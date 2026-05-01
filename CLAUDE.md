# gllm

**Inference Client** — High-level library for model management, scheduling, and engine orchestration.

## 🚨🚨🚨 铁律 ARCH-RUST-IS-CODEGEN — Rust = 代码生成器，推理阶段什么都不做

**Rust 的定位**: 代码生成器 + Hook ABI 工具库。**推理阶段 Rust 什么都不做。**

- ❌ **禁止** Rust 参与推理/计算/数据搬运/文本解码/采样/KV cache 管理/循环
- ❌ **禁止** 在热路径中出现 HashMap、String、Vec、for 循环、clone、malloc
- ❌ **禁止** 任何形式的 Rust 编排引擎（逐节点/逐层循环）
- ✅ **Rust 只做**: 模型加载时生成 JIT 机器码 → 推理时调用一次 JIT 入口函数
- ✅ **所有功能全部 JIT**: forward、采样、generate 循环、KV cache、stop condition、token→text、Guardrail、SG、HR、Intent、CoT、Early Exit、MoE — 全部编译为机器码
- ✅ **Hook**: JIT 内嵌条件 JMP。无 hook = 不生成跳转代码。Hook 通信 = 共享内存，不经过 Rust

**推理时 Rust 的全部操作**: 一次 CALL。返回后 output_buffer 中已有完整 UTF-8 文本。

## SPEC Location
- `./SPEC/` (Single Source of Truth, 28 documents, 142 REQs)
- `../gllm-kernels/SPEC/` (Backend constraints)

## 🚨 单工程双仓约定 (MONOREPO-PAIR)

**铁律：`gllm` 与 `../gllm-kernels` 是同一工程在两个 git 仓库中的切分，任何涉及"项目状态"、"工作区清洁"、"编码前提交"的规则，**必须同时覆盖两个仓库**，没有例外**。

- ✅ `gllm-kernels` 视为 `gllm` 的同项目组成部分（不是"第三方库"、不是"相关项目"）
- ✅ 编码任务开始前的 git 清洁检查必须同时在两个仓库执行（`gllm/` 和 `../gllm-kernels/`）
- ✅ 任何一个仓库存在未提交变更，两个仓库都算脏工作区
- ✅ SPEC 完善、代码开发、测试执行、E2E 验证一律按统一工程口径规划
- ❌ 禁止以"这是 gllm-kernels 仓库的问题，不在 gllm 任务范围内"为由绕过规则
- ❌ 禁止单独审计其中一个仓库就宣告完成

**理由**：`gllm` 上层逻辑与 `gllm-kernels` JIT codegen 的每一次变更都是强耦合的（SPEC 章节跨仓库引用、算子注册 ↔ lower/codegen ↔ client forward pass、telemetry offsets ↔ executor 读取）。分开看待会导致状态漂移、SSOT 断裂、孤岛模块。

## SPEC Index

| Document | Content | 状态 |
|----------|---------|------|
| `01-REQUIREMENTS.md` | 极化硅晶与通信墙生存阈值要求 (包含 TurboQuant 静态极化与 NUMA/PCIe/RDMA 硬件拓扑探测约束) + §18 Mega-Kernel Session/Multimodal/SG (REQ-MEGA-001~006) + §19 自动指令选择器 (REQ-AIS-001~004) + §20 MoE 算子 (REQ-MOE-001~002) | ✅ |
| `02-ARCHITECTURE.md` | 4层物理架构, Mega-Kernel 块级路由, TurboQuant, Epilogue 白嫖, 热修补, §13.12 硬件感知融合拓扑 (12 Profile: SM100+/SM90/SM80/SM70/AVX10.2/AVX10.1/AMX/AVX-512/AVX2/SME2/SVE2/NEON) | ⚠️ §1-§8 ✅, §9-§12/§14-§16 🟡, §13 ⚠️ |
| `03-DATA-STRUCTURE.md` | 全链路数据结构 (KV Cache, Paged Attention, HGAL, MoE, RDMA) | ✅ |
| `04-API-DESIGN.md` | 客户端公共 API (§7-§8 Semantic Gatekeeper 隐藏状态知识注入 SDK) | ✅ |
| `00-PHILOSOPHY.md` | 核心哲学原则 (Accuracy First, JIT Hot-Repair, No Pragmatic Hacks, ARCH-FULL-JIT) | ✅ |
| `01-JIT-PIPELINE.md` | JIT 四阶段管线 (Scalar→SymExec→IR→ISA Lowering) + CompilerGraph + DeviceProfile + §5.1 自动指令选择器 (ARCH-AUTO-INSTR-SELECT) + §3.4 TraceOp 扩展 (Compare/Cast/HReduce) | ✅ |
| `02-HARDWARE.md` | 硬件探测与 DeviceProfile 12 Profile (SM100+/SM90/SM80/SM70/AVX10.2/AVX10.1/AMX/AVX-512/AVX2/SME2/SVE2/NEON) | ✅ |
| `14-HW-INTRINSICS.md` | 硬件指令能力矩阵 (x86/ARM/NVIDIA/AMD/Apple GEMM+Attention+Dequant+Norm 指令 + 设备探测方法) | ✅ |
| `03-GRAPH-IR.md` | CompilerGraph IR 规范 + OpKind + SymDim + CallbackChain 契约 (pre_node/post_node/CallbackAction) | ✅ |
| `04-OPERATORS.md` | 算子族清单与注册规范 (GEMM/RmsNorm/RoPE/Attention/MoE/PLE/Gather/...) + §4.6 MoE (MoEGate/MoERouter/MoEDispatchPacked) | ✅ |
| `05-OPTIMIZATIONS.md` | 融合决策 + Epilogue 白嫖 + TileLevelFusion/ComputeRoot 寄存器约束 | ✅ |
| `06-RUNTIME.md` | 运行时执行模型 (FusedGraphExecutor, run_with_callbacks, KV cache 数据流) | ✅ |
| `07-LOADER.md` | 模型加载规范 (safetensors/GGUF/ONNX/PyTorch .bin + BF16→F32 并行化 + cache-blocked transpose §2.4 ARCH-LOADER-NORMALIZE + §2.5 MXFP4 分离格式 + §7.5 YAML Schema) | ✅ |
| `08-EXECUTOR.md` | Executor 规范 (§1.2.1 ARCH-KV-EFFECTIVE-MAXSEQ + §1.2.2 ARCH-KV-EFFECTIVE-LAYER SSOT 映射) | ✅ |
| `09-API.md` | 公共 API 契约（内部 trait/枚举/错误语义） | ✅ |
| `10-QUALITY.md` | 质量保证与数值对齐要求 | ✅ |
| `11-MODELS.md` | 支持的模型架构详细规范（全量索引） | ✅ |
| `SEMANTIC-GATEKEEPER.md` | **Semantic Gatekeeper 技术协议 (SSOT)** — Level Keys 预计算、Q-tap 截获、稳定性追踪、KnowledgeProvider trait、CallbackChain 集成、Mega-Kernel SgDetect/SgInject 共享内存、Callback Table (ABI arg 20) JIT 内 CALL 外部函数指针、E2E 验收 (REQ-SG-001..008) | ✅ REQ-SG-001~008 全部已实现; ✅ REQ-MEGA-SG-001 (hook_ctx_ptr + SgSharedMemory); ✅ REQ-MEGA-SG-002 (Callback Table + NativeCall + StackAlign + YMM save + SgDetect/SgInject FMA + KnowledgeProvider bridge + TextEncoder directional embedding); ✅ REQ-SG-008 complete: diff=true, 240 provider calls, Paris embedding injection changes argmax |
| `HEAD-ROUTING.md` | **Head Routing SDK 技术协议 (SSOT)** — 同一 generator LLM 多头 API (generate/classify_binary/classify_multiway/encode_to_layer) 运行时切换,零权重重载、零 JIT 重编译,E2E 验收 (REQ-HR-001..005) | ✅ |
| `GUARDRAIL.md` | **Guardrail SDK 技术协议 (SSOT)** — in-flight 安全 veto 探针,`attach_guardrail` + `GuardrailProbeCallback` (post_node),SafetyPolicy (HaltAndVeto/LogOnly/SampleDowngrade),正交于 SG/HR,E2E 验收 (REQ-GR-001..005) | ✅ |
| `INTENT.md` | **Intent Recall SDK 技术协议 (SSOT)** — `encode_intent` 截断前向至 anchor 层 pool hidden 作为意图识别向量,delegate 到 `encode_to_layer` (DRY),E2E 验收 (REQ-INTENT-001..003) | ✅ |
| `COT-REASONER.md` | **CoT Reasoner SDK (SSOT)** — 任意 LLM 原生 Chain-of-Thought 推理 (Manual + Auto 模式),Client 层 orchestration,零 Backend 扩展,Step Hook 推理引擎级思考控制 (REQ-COT-007..009),E2E 验收 (REQ-COT-001..009) | ✅ |
| `06-TESTING-STRATEGY.md` | 测试策略 (GGUF/ONNX/E2E/性能/观测/错误处理) | ✅ |
| `07-OBSERVABILITY.md` | Epilogue 白嫖遥测扩展, AbsolutePolicy 护栏, KvPageHeader 40B 设计 | ✅ |
| `ARCH-DATA-FLOW-CONTRACT.md` | 数据流唯一来源契约（gllm 侧 executor/loader 数据流 SSOT） | ✅ |
| `ARCH-DETAILED-DESIGNS.md` | ISV integration, quantized GEMM, GPU backend, adaptive chunking | ✅ |
| `P0-P3-ROADMAP.md` | Priority roadmap (all P0-P3 completed) | ✅ |
| `SUPPORTED_MODELS.md` | 20+ model architectures (generators/embeddings/rerankers) | ✅ |
| `12-STRATEGY-ARBITER.md` | 策略仲裁器 (InferenceMode Latency/Throughput, GraphProfile 模型图拓扑特征, StrategyBias 成本调制, HwOptEngine 集成) | ✅ |
| `DOCS/scheduling/jit-cache-protocol.md` | JIT 编译缓存协议 (三级缓存, SymDim 动态维度, CPU/GPU 统一) | ✅ |
| `DOCS/scheduling/ai-development-guideline.md` | 极简化内核执行底线开发思想原则 | ✅ |
| `DOCS/scheduling/hgal-scheduler-algorithm.md` | HGAL 调度算法规划基准 | ✅ |
| `../gllm-kernels/SPEC/ARCH-DATA-FLOW-CONTRACT.md` | **数据流唯一来源契约** — lower/executor 每个值的唯一数据源映射表，禁止独立计算/反推/硬编码 | ✅ |
| `../gllm-kernels/SPEC/GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md` | **全虚拟化编译管线 (SSOT)** — 元抽象: 编译时映射函数替代运行时物理操作; 十维全虚拟化图谱; VTC 虚拟 tensor; PDT 拓扑融合; 7 轮虚拟化求解; Phase 3 唯一物化点; §1.5.3 Session KV Cache 复用 + Multimodal Fused Hidden 注入 + SgDetect/SgInject 共享内存; §6.5/§6.6 Session/Multimodal 数据流契约; 后端无关 ABI 参数布局 (x86_64/AArch64/GPU) | 🟡 设计完成，实施中 |

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
> **gllm Principles**:
> 1.  **Accuracy > Throughput**: TurboQuant 2.0 通过数学级静态湮灭（而非运行时分支）保证量化精度。
> 2.  **Reliability First**: Memory management (PagedAttention) must have strict boundary checks and error recovery. Prefer OOM rejection over returning corrupted results. JIT must emit hardware Trap on Out-of-Bounds memory without CPU fallback.
> 3.  **JIT Hot-Repair & Block Routing (物理核内调度)**: 摒弃主机层面的复杂分流切换，一切交由运行时 JIT 发射的原子擦写指令 (Hot JMP Patching / DCE) 及根据 `SystemTopology` 切割的物理块式异构动态图处理。
> 4.  **NO PRAGMATIC HACKS (禁止务实方案)**: 所有实现必须遵循 SPEC 设计，追求最完美的架构。禁止使用 `unwrap_or(default)` 等 fallback、禁止硬编码魔法数字、禁止临时 workaround。宁可重构整个模块，也不接受技术债。
>
> **已废弃**: ARCH-ACCURACY-SCHED（规范序）、ARCH-ACCURACY-EXEC（串行微批次）、ARCH-ACCURACY-ISOLATION（阶段隔离）—— 三项均被 §9 Mega-Kernel 块级路由和 §10 Chunked Prefill 交织调度完全覆盖。

### 0.1 SymDim 动态维度铁律 (ARCH-SYMDIM-CODEGEN)
> **关联**: SPEC-archive/02-ARCHITECTURE.md §5.3, jit-cache-protocol.md
> **状态**: 🔴 IN PROGRESS (重构中)

**问题**: 当前 JIT codegen 假设所有维度在编译时已知（`m: usize`），导致动态 seq_len 需要 hack（硬编码 compile_seq_len=512 + 运行时 `[rbp+16]` 读取）。

**正确设计**:
1. **图层面**: 所有动态维度（seq_len、batch_size、total_seq）用 `SymDim::Symbolic("name")`
2. **JIT codegen**: 检测 `SymDim::Symbolic` 时，生成从 ShapeBinding 读取的通用循环
3. **执行时**: 通过 `ShapeBinding { "seq_len": actual_value }` 传递运行时值
4. **禁止**: `m.as_concrete().unwrap_or(512)` 等 fallback

**实现路径**:
- Phase 1: 所有 `build_*_graph` 的 seq_len 用 `SymDim::Symbolic`
- Phase 2: JIT codegen 支持 Symbolic（GEMM/elementwise/norm 全部）
- Phase 3: Executor 传递 ShapeBinding
- Phase 4: 清理所有 compile_seq_len hack

**承诺**: 完全符合 SPEC §5.3 铁律，零妥协。

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
- **FusedQkvNormRope** (Gemma 4): Q_proj/K_proj/V_proj + QkNorm + ValueNorm + DualRoPE 六算子融合
- HardwareFusionPass: 硬件不支持时降级到 Atomic
- DeadCodeElimination: 移除未使用节点

**CompilerGraph OpKind (Gemma 4 新增算子)**:
- `QkNorm { head_dim }`: Q/K 向量 L2 归一化 + √head_dim 缩放
- `ValueNorm { eps }`: V 向量无学习参数 RMSNorm
- `PerLayerEmbed { layer_idx, dim_per_layer }`: E2B/E4B 每层注入 token-identity + context-aware 信号
- `ColumnSlice { start, end }`: PLE 按层切片真实列切片 JIT 实现 (T37)
- `PatchEmbed` (Vision): 图像 → patch 序列 (Conv2D + Reshape) — 骨架中 (T44)
- `LearnedPos2D` (Vision): 2D 学习位置编码 (非 RoPE) — 骨架中 (T44)
- `DepthwiseConv1D` (Audio): Conformer 的深度卷积模块 — 骨架中 (T45)
- `RoPE { partial: f32 }` (扩展): DualRoPE 支持 partial=0.25 p-RoPE 路径

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

### 5.3 Supported Model Architectures (能力矩阵索引)

完整模型清单见 `SPEC/SUPPORTED_MODELS.md` 与 `SPEC/11-MODELS.md`。活跃架构能力状态:

| 架构 | CPU JIT | GPU codegen | 备注 |
|------|--------|-------------|------|
| Qwen3 / Qwen3MoE | ✅ | ✅ | 通用标杆 |
| Llama4 / SmolLM / InternLM | ✅ | ✅ | G-A 路径共用 |
| GLM-4 / GLM-5 | ✅ | ✅ | MoE (glm-4.7-flash) |
| Mistral3 / Ministral | ✅ | ✅ | Sliding Window |
| Phi4 / Phi4-mini | ✅ | ✅ | Partial RoPE |
| GptOss (gpt-oss-20b) | 🟡 模板待编写 | 🟡 | MoE + sliding/full attention 交替 + yarn RoPE + RMSNorm + SiLU + attention bias |
| DeepSeek V3/R1 / Kimi-K2 | ✅ | ✅ | MoE Router+SharedExperts |
| XLM-R / XLM-R-Next (Encoder) | ✅ | ✅ | Embedding / Rerank |
| **Gemma 4 (E2B/E4B/26B-A4B/31B)** | ✅ | ✅ | QkNorm + ValueNorm + DualRoPE + PLE(E2B/E4B) + SharedKvRef page 层 ✅; graph 层 🟡 T43; Vision/Audio 🟡 T44/T45; E2E 数值验证 🟡 T47 |

**Gemma 4 关键差异点** (T21-T42 接入完成):
- **DualRoPE**: sliding 层 θ=10K + partial=1.0;global 层 θ=1M + partial=0.25 (p-RoPE)
- **QkNorm / ValueNorm**: Q/K L2 归一化 + V 无学习参数 RmsNorm
- **PerLayerEmbedding** (仅 E2B/E4B): 每层 token-identity + context-aware 信号注入,经 `ColumnSlice` JIT 按层切片
- **SharedKvRef**: 后 `num_kv_shared_layers` 层不计算 K/V,按 page 引用 donor 层 KV (T39 page 层完成,T43 graph 层并行中)
- **FusedQkvNormRope**: pattern_fusion 识别 6 算子链 (T29/T41/T42)

## Directory Structure

```
src/
├── lib.rs                 # Library entry & public API re-exports
├── client.rs              # Client / AsyncClient (sync/async inference API)
├── classify.rs            # Classify API (sequence classification)
├── embeddings.rs          # Embeddings API
├── rerank.rs              # Rerank API
├── generation.rs          # Generation loop
├── tokenizer.rs           # Tokenizer integration
├── model_config.rs        # Tensor-driven model config (Ω1)
├── weight_loader.rs       # Weight loading utilities
├── weight_names.rs        # Canonical weight name mappings
├── kv_cache.rs            # KV Cache structures
├── quantization.rs        # Quantization support
├── static_compression.rs  # Static compression utilities
├── ffi.rs                 # C FFI exports (gllm_init/generate/destroy/client_version)
├── compat/                # gllm-kernels compatibility shim & forward passes
│   ├── mod.rs             # Re-exports gllm-kernels types (Backend, Element, etc.)
│   ├── types.rs           # Bridged type definitions
│   ├── scalar_ops.rs      # Scalar reference impls (Phase 0 only, NO runtime calls)
│   ├── jit_helpers.rs     # JIT compilation helpers
│   ├── weight_helpers.rs  # Weight tensor manipulation
│   ├── bert_forward.rs    # BERT/encoder forward pass
│   ├── decoder_forward.rs # Decoder (GPT-style) forward pass
│   ├── gpu_compile.rs     # GPU graph compilation & kernel launch
│   ├── gpu_helpers.rs     # GPU memory & transfer utilities
│   ├── memory.rs          # Memory management abstractions
│   ├── cpu_backend.rs     # CPU inference backend
│   ├── cuda_backend.rs    # CUDA inference backend
│   ├── hip_backend.rs     # HIP (ROCm) inference backend
│   └── metal_backend.rs   # Metal inference backend
├── loader/                # Model fetching & parsing
│   ├── mod.rs             # Unified loading entry (auto format detection)
│   ├── hf_hub.rs          # HuggingFace Hub downloader
│   ├── modelscope.rs      # ModelScope downloader
│   ├── downloader.rs      # Download orchestration (HF→MS fallback)
│   ├── format_detector.rs # Auto format detection
│   ├── safetensors.rs     # SafeTensors parser (zero-copy)
│   ├── adapter.rs         # 类型桥接层 (DType/KernelTensorView/GgufAdapter/QuantType 映射)
│   ├── parallel.rs        # Parallel layer loading
│   ├── pytorch.rs         # PyTorch format support
│   ├── gguf/              # GGUF parser (zero-copy, Ω1 compliant)
│   │   ├── mod.rs
│   │   ├── reader.rs      # GGUF file reader
│   │   ├── slice.rs       # Zero-copy tensor slicing
│   │   └── types.rs       # GGUF type definitions
│   └── onnx/              # ONNX protobuf parser (prost, graph pattern matching)
│       ├── mod.rs
│       ├── model.rs       # ONNX model loading
│       ├── attributes.rs  # ONNX attribute parsing
│       ├── types.rs       # ONNX type mappings
│       ├── external.rs    # External data loading
│       ├── pack.rs        # Tensor packing
│       ├── tests.rs       # ONNX parser tests
│       └── tensor/        # Tensor parsing
├── arch/                  # Architecture YAML templates → OnnxGraph
│   ├── mod.rs             # Template registry
│   ├── registry.rs        # Architecture registry
│   ├── resolve.rs         # Architecture resolution from metadata
│   ├── template.rs        # YAML → OnnxGraph parser
│   └── templates/         # YAML 模板目录 (build.rs 扫描自动注册,SSOT)
│                          # 新增架构 = 扔 YAML 进此目录,无需改任何 Rust 代码
├── graph/                 # DAG optimizer (unified representation)
│   ├── mod.rs
│   ├── types.rs           # OnnxGraph extended types
│   ├── executor.rs        # FusedGraph executor
│   └── optimizer/         # Optimization passes
│       ├── mod.rs
│       ├── pass.rs              # Pass trait & registry
│       ├── pattern_fusion.rs    # Pattern-based fusion (FlashAttn, SwiGLU, etc.)
│       ├── hardware_fusion.rs  # Hardware-aware fusion/degradation
│       ├── constant_folding.rs # Constant folding pass
│       └── dead_code.rs        # Dead code elimination
├── engine/                # Execution engine (wraps gllm-kernels)
│   ├── mod.rs
│   ├── executor.rs        # Executor (batch orchestration)
│   └── pipeline.rs        # Pipeline management
├── scheduler/             # Batching & KV Cache management (HGAL)
│   ├── mod.rs
│   ├── paged_scheduler.rs # PagedAttention core
│   ├── batcher.rs         # Continuous Batching
│   ├── hgal.rs            # HGAL scheduler algorithm
│   ├── allocator.rs       # Page allocator
│   ├── memory_manager.rs  # GlobalMemoryManager
│   ├── prefix_index.rs    # KvPrefixIndex (trie-based)
│   ├── sequence.rs        # SequenceGroup
│   ├── types.rs           # Scheduler types
│   ├── observer.rs        # RuntimeObserver (metrics & tracing)
│   ├── policy.rs          # JIT scheduling policies (hot-switchable)
│   ├── jit_types.rs       # JIT type definitions
│   └── vllm2024.rs        # SwiftKV / Scheduler2024Config / AdaptiveChunkPolicy
├── backend/               # Backend detection (ARCH-ZERO-FALLBACK: no OOM fallback)
│   ├── mod.rs
│   ├── detection.rs       # Auto-detect CUDA→ROCm→Metal→CPU
│   └── (fallback.rs deleted — ARCH-ZERO-FALLBACK)
├── manifest/              # Model manifest types
│   ├── mod.rs
│   └── types.rs
└── bin/                   # Binary utilities
    ├── download.rs        # Model download CLI
    └── debug_shape.rs     # Shape debugging tool

examples/                   # Example programs
├── debug_gguf_kv.rs       # GGUF KV cache debugging
├── debug_kv9_after.rs     # KV9 debugging
├── debug_qwen3_tensors.rs # Qwen3 tensor debugging
├── list_gguf_keys.rs      # List GGUF keys
├── list_gguf_tensors.rs   # List GGUF tensors
├── list_qwen3_tensors.rs  # List Qwen3 tensors
├── parse_gguf_string_offset.rs  # Parse GGUF string offset
├── parse_gguf_string_table.rs   # Parse GGUF string table
├── read_gguf_direct.rs    # Read GGUF directly
├── read_gguf_strings.rs   # Read GGUF strings
├── read_gguf_tokenizer.rs # Read GGUF tokenizer
├── read_tokenizer_simple.rs # Simple tokenizer reader
├── regression.rs          # Regression tests
├── test_gguf_metadata.rs  # Test GGUF metadata
├── test_gguf_reader.rs    # Test GGUF reader
└── test_gguf_tokenizer.rs # Test GGUF tokenizer
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

## 🚨 编译时 dtype 感知 — 原生混合精度 (ARCH-DTYPE-JIT-TYPED)

**铁律：dtype 从模型 TensorMeta 自动推断，JIT 编译时根据 dtype 生成特化机器码，运行时零分支。**

### 核心规则

1. ✅ **dtype 在 JIT 编译时已知**：模型加载时每个 tensor 的 dtype 已知（TensorMeta.dtype），JIT 编译时读取并影响指令选择
2. ✅ **编译时单态化**：对每种 dtype 组合生成特化机器码（BF16→VDPBF16PS, F32→VMULPS），运行时零 dtype 分支
3. ✅ **双路径共存**：原生混合精度（模型原始 dtype）+ TurboQuant（额外量化格式），两种模式互不冲突
4. ✅ **自动类型推断**：TraceOp body 通过 TypedSlot 携带 dtype，auto_select 根据 dtype 选择 VmInstr

### 禁止规则

1. ❌ **禁止硬编码 `QuantPrecision::F32`**：dtype 必须从 `graph.tensor(op.inputs[0]).dtype` 推断，不得硬编码
2. ❌ **禁止 `computation_elem_bytes()` 硬编码 F32**：必须用 `op_input_dtype(op, graph).elem_bytes()`
3. ❌ **禁止硬件不支持时 fallback 到 F32**：必须返回 Error
4. ❌ **禁止运行时 `match dtype` 分支**：dtype 在编译时 bake 进机器码，运行时零分支

### 审计命令

```bash
# 硬编码 QuantPrecision::F32（仅 op_input_dtype 默认值例外）
grep -rn "QuantPrecision::F32" ../gllm-kernels/src/compiler/codegen/vm/plan_lower.rs | grep -v "op_input_dtype\|unwrap_or"

# computation_elem_bytes 已废弃
grep -rn "computation_elem_bytes" ../gllm-kernels/src/compiler/codegen/vm/

# row_stride_bytes 已废弃
grep -rn "row_stride_bytes" ../gllm-kernels/src/compiler/codegen/vm/
```

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

**单实例强制要求（ARCH-SINGLE-MODEL-INSTANCE）**：
- ❌ **禁止同时加载超过 1 个模型实例运行测试**
- ❌ 禁止在后台跑一个 E2E 测试的同时，前台又启动另一个 E2E 测试
- ❌ 禁止通过多个 `Bash` 调用并行启动不同模型的 E2E 测试
- ✅ 同一时刻只能有 **一个** 模型加载在内存中
- ✅ 上一个测试完成后才能启动下一个
- ✅ `cargo test --test test_e2e_generator -- --test-threads=1` 自动串行

**理由**：
- 大模型（GPT-OSS 20B）仅权重就占 ~40GB 内存
- 同时加载两个模型会导致 OOM 或 swap thrashing
- 系统物理内存有限，单实例已接近上限
- E2E 测试涉及真实模型下载、文件 I/O、CPU 推理

**运行命令**：
```bash
# 正确：E2E 测试单线程运行（一次只跑一个）
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

## 🚨 禁止孤岛模块 (NO_ISLAND_MODULE)

**铁律：新增模块必须验证真实调用链接入，禁止"编译通过+测试通过=完成"**：

**问题**：新功能实现为独立模块（新文件+新结构体+新算法+单元测试），编译通过、测试通过，但真实推理路径完全没调用它——仍然走默认值/旧路径。模块成为自洽但无用的孤岛。

**OnceLock/Default 陷阱**：`Default::default()` 和 `OnceLock::get_or_init(|| default)` 会静默吞掉"没接入"的事实。模块没被调用时系统仍然正常工作，只是用了中性默认值。

**强制要求**：

1. **SPEC 必须包含 Integration Trace 章节**：从用户 API 入口到最终消费点的完整调用链，标明每一跳的 `文件:函数:行号`。不是"应该在这里调用"，而是"在 client.rs:build_state():L215 调用 GraphProfiler::profile()"。

2. **实现完成后必须 grep 验证**：
   ```bash
   # 验证新函数在非测试代码中被调用
   grep -rn "StrategyArbiter::arbitrate" src/ | grep -v test | grep -v "mod tests"
   # 如果结果为空 → 没接入 → 未完成
   ```

3. **集成测试验证行为差异**：必须写一个测试证明"不同输入 → 不同输出"通过真实路径传导。单元测试验证组件正确性，集成测试验证组件被真实消费。

- ❌ 禁止以"编译通过+单元测试通过"为完成标准
- ❌ 禁止新模块的公共函数仅在 `#[cfg(test)]` 中被调用
- ❌ 禁止依赖 `Default::default()` 掩盖"没接入"
- ✅ 新模块的核心函数必须在非测试代码路径中至少有一个调用点
- ✅ 调用点必须在真实推理/加载路径上（不是 dead code）
- ✅ 集成测试必须证明模块输出影响了最终行为

**审计方法**：
```bash
# 对每个新增的 pub fn，验证非测试调用:
grep -rn "函数名" src/ --include="*.rs" | grep -v "#\[cfg(test)\]" | grep -v "mod tests" | grep -v "///"
```

**理由**：孤岛模块是最隐蔽的 bug——系统"正常工作"但新功能完全无效。编译器和测试框架都无法检测到这种问题，只有调用链追踪能发现。

## 🚨 硬件静态分派禁止降级 (NO_HW_DEGRADATION)

**铁律：硬件参数在 `DeviceProfile::detect()` 时一次性探测、运行时固定。JIT codegen 必须为当前硬件生成最优路径代码，禁止任何形式的硬件降级/退化**：

- ❌ `HardwareFusionPass` 将融合算子降级为原子算子（如 `FlashAttention → Atomic("Attention")`）
- ❌ `FusedOp::SwiGLU → FusedOp::Atomic("SwiGLU")` 因为 CPU 没有 AVX2
- ❌ `FusedOp::MoERouting → Atomic` 因为 GPU SM < 8.0
- ❌ `supports_*()` 函数返回 false → 拆分融合图为独立子操作
- ❌ "硬件能力不足所以降级" 作为拆分融合算子的理由
- ✅ CPU 没有 FlashAttention 硬件 → codegen 生成 cache-friendly tiled attention（仍然是融合的）
- ✅ SM70 没有 tensor core gating → codegen 生成 wmma 路径的 MoERouting（仍然融合）
- ✅ 标量 CPU → codegen 生成标量循环的 SwiGLU（仍然融合，不是拆成 3 个独立 op）
- ✅ 硬件差异体现在 **codegen 层的指令选择**（SIMD 宽度、寄存器分配、分块策略），不是在 **fusion 层的算子拆分**

**架构原则**：
```
错误: FusionRule 看到硬件能力不足 → 降级为 Atomic → executor 拆分执行
正确: FusionRule 生成融合图 → JIT codegen 根据 DeviceProfile 为每种硬件生成最优机器码
```

**理由**：硬件参数是编译时常量。降级意味着放弃优化，而正确做法是在 codegen 层实现该硬件的最优路径。降级是偷懒——"我不会写 SM70 的 MoERouting，所以把它拆开"——这违反了"功能缺失必须补全实现"的铁律。

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

## 🚨 自动指令选择器铁律 (ARCH-AUTO-INSTR-SELECT)

**正确架构（类似 LLVM SelectionDAG）**：
```
Scalar → SymExec → TraceOp → [自动指令选择] → VmInstr → ISA Lowering → Machine Code
                                    ↑
                               这一步必须是自动的
                               (类似 LLVM SelectionDAG)
                               由 ComputePattern 驱动，算法保证正确性
```

**问题根因**：`gllm-kernels/src/compiler/codegen/vm/plan_lower.rs` 中 `emit_standalone_op` 的 50+ 手写 `OpKind::*` match arm 是 JIT 层 bug 的系统性源头。每个 arm 手动管理寄存器分配、指针算术、内存布局，极易引入堆栈/寄存器/内存偏移错误。

**铁律**：

1. ❌ **禁止手写 TraceOp → VmInstr 映射**：必须使用 `auto_select.rs` 中的 `auto_lower_trace()` 查表法，每个 TraceOp 变体自带 VmInstr 映射语义
2. ❌ **禁止手写 OpKind → VmInstr match arm**：`emit_standalone_op` 必须基于 `ComputePattern` 自动分发，不允许逐个 OpKind 手写 lowering
3. ❌ **禁止 opaque 算子跳过 trace**：所有 OpKind 必须在 `ScalarOpRegistry` 注册 scalar 参考实现，SymExec 必须能提取 trace
4. ❌ **禁止创建 per-OpKind 手写函数**：不允许创建 `emit_meanpool_auto` / `emit_rope_auto` / `lower_layernorm` / `lower_rope_full` / `lower_meanpool` 等按 OpKind 命名的 VmInstr 发射函数。每种 ComputePattern 只有**一个**通用处理器
5. ❌ **禁止以"结构型算子"为由绕过 auto_select**：Gather/Attention/MoE 等结构型算子必须通过扩展 TraceOp 语义纳入自动指令选择，不允许保留独立手写 lowering 函数
6. ✅ **TraceOp 语义扩展是第一选择**：遇到任何算子无法被现有 TraceOp 表达时，必须先扩展 TraceOp 语义（增加新变体），而非绕过 auto_select 手写 VmInstr。新增 TraceOp 变体只需在 `auto_select.rs` 的 `dispatch_trace_op` 添加一个 match arm — 这是架构设计允许且鼓励的操作，不需要特殊审批
7. ✅ **Elementwise/BinaryElementwise**：`try_auto_dispatch_by_pattern()` → `auto_lower_trace()` 自动完成
8. ✅ **Injective（多输入多输出逐元素）**：`emit_injective_inline` + `auto_lower_trace_multi`，覆盖 RoPE 等所有多输出逐元素变换
9. ✅ **Reduction（归约+归一化）**：`emit_reduction_inline` + `auto_lower_trace`，覆盖 MeanPool/L2Normalize/Argmax 等所有归约算子
10. ✅ **NormLike（三阶段归一化）**：`emit_normlike_inline` + `auto_lower_trace` / `auto_lower_trace_multi`，覆盖 RmsNorm/LayerNorm/QkNorm/ValueNorm
11. ✅ **Gemm**：`emit_gemm_inline_with_hook`，硬件分块策略
12. ✅ **Structural（Gather/ColumnSlice/Attention/MoE）**：必须通过 TraceOp 扩展纳入 auto_select，不允许独立手写 lower_* 函数。结构型算子的索引计算、行复制、循环嵌套全部用 TraceOp 语义表达
13. ✅ **新增算子只需**：注册 scalar impl → SymExec 提取 ComputePattern → 自动路由到对应通用处理器，零额外代码

**TraceOp 语义扩展原则**：
- 测试发现语义不足 = TraceOp 需要扩展，这是正常的架构演进
- 任何 VmInstr 都有对应的 TraceOp 语义表达（包括 ScalarLoad/VecLoad/IntMulStride/LoadPtr 等结构型指令）
- 新增 TraceOp 变体的流程：(1) 在 `trace.rs` 添加枚举变体 (2) 在 `auto_select.rs` 添加 match arm (3) 在 `verify.rs` 确保 def-before-use 覆盖
- **禁止以"后端约束"为由拒绝扩展 TraceOp**：后端（x86/ARM/GPU）差异由 ISA Lowering 层处理，TraceOp 是硬件无关的中间表示

**实现状态**：
- Phase 1 (`auto_select.rs`): ✅ `auto_lower_trace()` / `auto_lower_trace_raw()` / `auto_lower_trace_multi()` — TraceOp → VmInstr 自动查表
- Phase 2 (ComputePattern dispatch): ✅ elementwise 自动分发 + Reduction/Injective/NormLike/Gemm 通用处理器
- Phase 3 (TraceOp 扩展): ✅ Compare/Cast/HReduce/ConditionalBranch 全部已实现
- Phase 4 (结构算子 TraceOp 化): 🟡 进行中 — Gather 需要扩展 TraceOp 语义（LoadIndexed/StoreIndexed），Attention/MoE 待覆盖
- Phase 5 (手写 lowering 清除): ❌ `lower_gather`/`lower_mha_with_hook`/`lower_moe_*` 仍为手写，需迁移到 auto_select

**核心验证标准**：整图融合后的 JIT 机器码中，堆栈、寄存器分配、内存布局错误必须为零。通过符号执行 + 自动指令选择从根本上保证正确性，而非逐个修具体 bug。

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
| C1 | `src/backend/fallback.rs` | rerank `unwrap_or(0.0)` → 返回错误; **文件已物理删除 (ARCH-ZERO-FALLBACK)** | gllm `654642b` |
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

## 🚨 SymDim 穿透铁律 — 禁止降级为编译时常量 (ARCH-SYMDIM-NO-CONST-DEGRADE)

**铁律：SymDim::Symbolic 维度在整个管线中必须保持 Symbolic 语义，禁止在任何环节降级为编译时常量。**

### 核心禁止规则

1. ❌ **禁止 `max_for_allocation()` 作为运算参数**：`max_for_allocation()` 仅用于 buffer 分配大小计算（确保内存安全上界），不得用于循环 bound、元素计数、维度推导等影响计算逻辑的场景
2. ❌ **禁止 `BoundExpr::Const` 处理 Symbolic 维度**：当维度来自 `SymDim::Symbolic` 时，循环 bound 必须使用 `BoundExpr::Symbolic`，从运行时参数（如 `[rbp+16]` 的 seq_len）读取实际值
3. ❌ **禁止 `infer_elem_count` 返回固定值**：`infer_elem_count` 遇到 Symbolic 维度时必须返回 Symbolic 信息（如 `(SymDim, usize)` 元组），不得用 `max_for_allocation` 压平
4. ❌ **禁止 executor 层手动计算 seq_len**：`activation_elems / feature_dim` 这种从字节数反推维度的 hack 是硬编码。seq_len 必须从图的 SymDim 元数据正向传递到执行层
5. ❌ **禁止编译时固定 `compile_seq_len`**：JIT 编译不应依赖编译时固定的 seq_len。GEMM 已正确使用 `BoundExpr::Symbolic`，所有其他算子（elementwise、norm、attention、gather）必须同等对待
6. ❌ **禁止 `SYMDIM_MAX_SEQ_LEN` 作为运算参数**：该常量仅用于 `SymDim::Symbolic { max_value }` 的 buffer 分配上界，不得出现在 `output_numel / SYMDIM_MAX_SEQ_LEN` 等除法运算中

### 正确架构

```
SymDim::Symbolic("seq_len") → BoundExpr::Symbolic(SymBound) → JIT 循环从 [rbp+16] 读运行时值
SymDim::Concrete(384) → BoundExpr::Const(384) → JIT 循环用编译时常量（仅限内层非 Symbolic 维度）
```

- ✅ GEMM M 维度已正确使用 `BoundExpr::Symbolic`（参考实现）
- ✅ Norm 外层 seq 循环已正确使用 `BoundExpr::Symbolic`
- ❌ Elementwise 必须跟进：外层 seq 维度用 Symbolic，内层 feature 维度用 Const
- ❌ Attention/MHA 必须用 `emit_loop`，禁止 Rust for 循环展开 VmInstr

### 审计命令

```bash
# 检查 max_for_allocation 是否用于非 buffer 分配场景
grep -rn "max_for_allocation" src/ gllm-kernels/src/ | grep -v "alloc\|buffer\|capacity\|size_bytes\|total_bytes"

# 检查 BoundExpr::Const 是否处理了 Symbolic 维度
grep -rn "BoundExpr::Const" gllm-kernels/src/compiler/codegen/ | grep -v test

# 检查 SYMDIM_MAX_SEQ_LEN 是否用于运算
grep -rn "SYMDIM_MAX_SEQ_LEN" src/ | grep -v "max_value\|buffer\|alloc\|capacity"
```

## 🚨 JIT 循环结构铁律 — 禁止 Rust 循环展开 VmInstr (ARCH-NO-LOOP-UNROLL)

**铁律：JIT codegen 中所有数据相关的循环必须使用 `emit_loop()`（LoopBegin/LoopEnd VM 指令），禁止用 Rust `for` 循环展开为扁平 VmInstr 序列。**

### 问题根因

MHA lower 中 `for h in 0..num_heads { for qi in 0..seq_len { ... } }` 将 seq_len²×heads 次迭代展开为百万级 VmInstr，导致：
- RegAllocator InterferenceGraph O(n²) 编译时间爆炸（seq=64 → 1.4M 条指令 → 不可能完成）
- VmProgram 内存占用巨大
- 优化 pass 效率极低

### 核心规则

- ❌ **禁止 `for qi in 0..seq_len` 展开 VmInstr**：seq_len 是运行时维度，不能在编译时展开
- ❌ **禁止 `for h in 0..num_heads` 展开 VmInstr**：head 循环也必须用 `emit_loop`（除非 num_heads ≤ 4 的微展开优化）
- ✅ **必须使用 `prog.emit_loop(BoundExpr, step, |prog, ctr, off| { ... })`**：所有数据维度循环
- ✅ **Const bound 仅用于编译时确定且极小的维度**（如 head_dim 的向量化内循环 dim/8 = 4 次）

### 合法的 Const 展开

仅以下场景允许 `BoundExpr::Const`：
1. 编译时确定的**内层**微维度（head_dim/lanes, nr 微核列数）
2. 常量 ≤ UNROLL_THRESHOLD (4) 的完全展开优化
3. 非数据维度的结构循环（如 epilogue trace 中的 op 数量）

## 🚨 Gather 必须走 JIT 管线 (ARCH-GATHER-JIT)

**铁律：Gather（embedding lookup）是索引计算操作，必须在 JIT 中实现，禁止绕过 JIT 走 CPU。**

- ❌ **禁止 `if op.name() == "Gather" { execute_gather_cpu(...); continue; }` 模式**
- ❌ **禁止在 executor 层对特定 OpKind 做特殊分支处理**
- ✅ **Gather 的 lower 必须生成正确的 VM 指令**：从 input_ptr 读 index → 计算 table 偏移 → 复制行到 output
- ✅ **需要的 VM 指令扩展**：标量 f32→int 转换 + 标量乘法（计算行偏移）
- ✅ **如果 VM 指令集不支持**：先扩展 VM 指令集，再实现 Gather lower

### 模型配置参数禁止硬编码

- ❌ **禁止 `(i + 2) as f32` 式的 position offset**：RoBERTa position_offset 必须从模型配置读取
- ❌ **禁止 `vec![0.0; seq_len]` 式的 token_type_ids**：必须从 tokenizer 或模型配置获取
- ✅ 所有模型特定参数（position_offset, pad_token_id, bos_token_id）从 `ModelConfig` / `ModelManifest` 读取

## 技能索引

| 技能 | 描述 |
|------|------|
| `/root-cause-debugging-philosophy` | 高维度根本性根治调试哲学与SPEC驱动开发 |
| `/concurrent-agent-orchestration` | 并发子agent编排：波次调度、冲突避让、用量限额恢复策略 |
| `/e2e-test-debugging-workflow` | E2E测试调试与数值对齐工作流 |
| `/jit-pipeline-development` | JIT编译管线开发规范与算子补充流程 |
