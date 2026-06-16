# gllm

**Inference Client** — High-level library for model management, scheduling, and engine orchestration.

## SPEC Location
- `./SPEC/` (SSOT, 76 documents, ~1350 REQs)
- `../gllm-kernels/SPEC/` (JIT codegen + VmInstr SSOT)
- `../gllm-nccl/SPEC/` (Distributed communication, 10 documents)

## SPEC Index

| Document | Content | 状态 |
|----------|---------|------|
| `00-PHILOSOPHY.html` | 核心哲学 (Accuracy First, JIT Hot-Repair, No Pragmatic Hacks, ARCH-FULL-JIT) | ✅ |
| `01-REQUIREMENTS.html` | 生存阈值 + Mega-Kernel Session/Multimodal/SG + 自动指令选择器 + MoE 算子 | ✅ |
| `02-ARCHITECTURE.html` | 4层物理架构, Mega-Kernel 块级路由, TurboQuant, 12 Profile 硬件感知融合拓扑 | ✅ |
| `03-DATA-STRUCTURE.html` | 全链路数据结构 (KV Cache, PagedAttention, HGAL, MoE, RDMA) | ✅ |
| `04-API-DESIGN.html` | 客户端公共 API + Semantic Gatekeeper SDK | ✅ |
| `01-JIT-PIPELINE.html` | JIT 四阶段管线 + CompilerGraph + DeviceProfile + 自动指令选择器 + TraceOp 扩展 | ✅ |
| `02-HARDWARE.html` | 硬件探测与 DeviceProfile 12 Profile | ✅ |
| `03-GRAPH-IR.html` | CompilerGraph IR + OpKind + SymDim + CallbackChain | ✅ |
| `04-OPERATORS.html` | 算子族清单与注册规范 + MoE | ✅ |
| `05-OPTIMIZATIONS.html` | 融合决策 + Epilogue 白嫖 + TileLevelFusion/ComputeRoot | ✅ |
| `06-RUNTIME.html` | 运行时执行模型 + KV cache 数据流 | ✅ |
| `07-LOADER.html` | 模型加载规范 (safetensors/GGUF/ONNX/PyTorch + BF16→F32 并行化) | ✅ |
| `08-EXECUTOR.html` | Executor 规范 (ARCH-KV-EFFECTIVE-MAXSEQ/LAYER SSOT) | ✅ |
| `09-API.html` | 公共 API 契约 | ✅ |
| `10-QUALITY.html` | 质量保证与数值对齐 | ✅ |
| `11-MODELS.html` | 支持的模型架构详细规范 | ✅ |
| `12-STRATEGY-ARBITER.html` | 策略仲裁器 | ✅ |
| `14-HW-INTRINSICS.html` | 硬件指令能力矩阵 | ✅ |
| `15-JIT-CONTEXT.html` | 统一 JIT 编译上下文 (JitContext 资源管理) | ✅ |
| `15-GPU-HOST-GLUE.html` | GPU Mega-Kernel Host 执行胶水 | ✅ |
| `16-DEVICE-FUSION.html` | 设备特化融合 PASS | ✅ |
| `17-DEVICE-CODEGEN.html` | 设备特化 JIT codegen | ✅ |
| `18-SYMDIM-PAGED-KV.html` | SYMDIM 动态维度 + PagedAttention 集成 | ✅ |
| `19-KV-CACHE-OPTIMIZATION.html` | KV Cache 智能优化 (动态稀疏 + 混合精度 + 页级表达) | ✅ |
| `20-BATCH-CONCURRENT-INFERENCE.html` | 批量并发推理 M 维度统一架构 | ✅ |
| `21-WEIGHT-PAGING.html` | 权重分页统一 | ✅ |
| `22-PAGE-COMPRESSION.html` | 页级压缩 + 三级换入换出 | ✅ |
| `23-QUANT-CODEGEN-ALGO.html` | 全量化格式算法化 JIT 生成器 (22 种 QuantType) | ✅ |
| `24-QUANT-PIPELINE-JIT.html` | 量化算子 JIT 管线化 | ✅ |
| `27-ALGORITHM-TEMPLATE.html` | 算法模板声明式表达 | ✅ |
| `28-GRAPH-RESOURCE-PLANNER.html` | 全图资源预规划 | ✅ |
| `31-EXECUTOR-DECOMPOSITION.html` | Executor 分解架构 (6 Coordinator) | ✅ |
| `32-MEGA-KERNEL-ENHANCEMENT.html` | Mega-Kernel 性能增强 (Prefill/Decode 差异化 + SM 分区 + 自治批调度) | ✅ |
| `33-MLA-MATRIX-ABSORPTION.html` | MLA 矩阵吸收 (DeepSeek V3/R1/Kimi-K2) | ✅ |
| `34-MTP-MULTI-TOKEN-PREDICTION.html` | MTP 多 Token 预测 | ✅ |
| `35-QUANTIZED-WEIGHT-PAGE-COMPRESSION.html` | 量化原生权重页压缩 | ✅ |
| `36-GLLM-WEIGHT-FORMAT.html` | gllm 原生模型权重分发格式 (.gllm) | ✅ |
| `39-UNIFIED-MEGA-KERNEL.html` | **统一编译器架构 (SSOT)** — 编译器=喂什么编译什么; 唯一编译入口 `compile()`; 所有图走同一管线产出 `MegaKernelFn`; `is_encoder`/`needs_kv_for_decode` 已删除; `GraphTopologyAnalysis` 替代; Phase 3 按融合组顺序 emit; `GroupMarker` 替代硬编码层循环; §7 编译管线三层并行化 | 🔄 Phase 3 实施中 |
| `40-END-TO-END-DATA-FLOW.html` | **端到端数据流 (SSOT)** — BUILD vs COMPILE 边界; MegaKernelFn 22 参数 ABI; GraphTopologyAnalysis 拓扑推导; OutputMode 裁剪 | ✅ |
| `SEMANTIC-GATEKEEPER.html` | Semantic Gatekeeper SDK (REQ-SG-001..008) | ✅ |
| `HEAD-ROUTING.html` | Head Routing SDK (REQ-HR-001..005) | ✅ |
| `GUARDRAIL.html` | Guardrail SDK (REQ-GR-001..005) | ✅ |
| `INTENT.html` | Intent Recall SDK (REQ-INTENT-001..003) | ✅ |
| `COT-REASONER.html` | CoT Reasoner SDK (REQ-COT-001..009) | ✅ |
| `INTENT-TRACKER.html` | Signal-Aware Intent Tracker (REQ-SIT-001..009) | ✅ |
| `06-TESTING-STRATEGY.html` | 测试策略 | ✅ |
| `07-OBSERVABILITY.html` | Epilogue 白嫖遥测 + KvPageHeader | ✅ |
| `ARCH-DATA-FLOW-CONTRACT.html` | 数据流唯一来源契约 (gllm 侧) | ✅ |
| `ARCH-DETAILED-DESIGNS.html` | ISV/量化GEMM/GPU/自适应chunking | ✅ |
| `SUPPORTED_MODELS.html` | 20+ 模型架构 (generator/embedding/reranker) | ✅ |
| `DOCS/scheduling/jit-cache-protocol.html` | JIT 编译缓存协议 (三级缓存, SymDim 动态维度) | ✅ |
| `DOCS/scheduling/ai-development-guideline.html` | 极简化内核执行底线开发原则 | ✅ |
| `DOCS/scheduling/hgal-scheduler-algorithm.html` | HGAL 调度算法规划基准 | ✅ |
| `DOCS/architecture/gemma4-altup.html` | AltUp + PLE 技术协议 (SSOT) | 🔄 代码实施中 |
| `DOCS/architecture/nvfp4-technical-reference.html` | NVFP 技术参考 | ✅ |
| `DOCS/architecture/awq-gptq-technical-reference.html` | AWQ/GPTQ 技术参考 | ✅ |
| `DOCS/architecture/iq-kquant-decode-reference.html` | IQ + K-Quant 低比特解码参考 | ✅ |
| `DOCS/architecture/gpu-ptx-codegen-reference.html` | GPU PTX Codegen 参考 | ✅ |
| `DOCS/architecture/longctx-moe-swa-reference.html` | 长上下文/稀疏注意力/MoE 共享专家 | ✅ |
| `DOCS/architecture/2026-frontier-reference.html` | 2026 前沿推理系统技术洞察 | ✅ |
| `DOCS/architecture/fwht-mla-pagedaddr-reference.html` | FWHT/MLA/PagedAttention 寻址参考 | ✅ |
| `../gllm-kernels/SPEC/GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.html` | 全虚拟化编译管线 (SSOT) + §0.8 dtype 感知编译 (REQ-DTYPE-001~008) | ✅ |
| `../gllm-kernels/SPEC/ARCH-DATA-FLOW-CONTRACT.html` | 数据流唯一来源契约 (lower/executor 每个值的数据源映射) | ✅ |
| `../gllm-kernels/SPEC/25-JIT-LIFECYCLE-INFRASTRUCTURE.html` | JIT 管线声明式生命周期管理 (REQ-LC-001~012) | ✅ |
| `../gllm-kernels/SPEC/26-VMINSTR-RATIONALIZATION.html` | VmInstr 去类型化合并 + 缺失指令补全 (REQ-VR-001~014) | ✅ |
| `../gllm-kernels/SPEC/37-HARDWARE-ACCELERATION.html` | 硬件加速全面集成 (REQ-HWACC-001~012) | ✅ |
| `../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.html` | 分布式 KV Cache 多层分页 (REQ-DP-001~014) | ✅ |

---

## 🚨 铁律索引

以下铁律按优先级排列，AI 行为必须严格遵守：

1. [JIT 数据顺从宪法](#-铁律-arch-jit-data-yields--jit-代码顺从数据而非数据顺从代码)
2. [JIT 顺从宪法](#-铁律-arch-jit-yields--jit-代码顺从硬件输入配置参数而动)
3. [三仓约定](#-三仓约定-triple-repo-pair)
4. [BUILD vs COMPILE 边界](#-build-vs-compile-边界铁律-arch-build-compile-boundary)
5. [Rust = 代码生成器](#-铁律-arch-rust-is-codegen--rust--代码生成器推理阶段什么都不做)
6. [统一编译器架构](#-铁律-arch-unified-compiler--统一编译器不假设图结构)
7. [自动指令选择器](#-铁律-arch-auto-instr-select--自动指令选择器)
8. [JIT 编译管线](#-铁律-jit-编译管线)
9. [dtype 感知](#-铁律-arch-dtype-jit-typed--编译时-dtype-感知)
10. [SymDim 穿透](#-铁律-arch-symdim-no-const-degrade--symdim-穿透禁止降级为编译时常量)
11. [禁止逐层展开](#-铁律-no-layer-expand--禁止逐层展开)
12. [禁止 JIT 循环展开](#-铁律-arch-no-loop-unroll--禁止-rust-循环展开-vminstr)
13. [禁止硬件降级](#-铁律-no-hw-degradation--硬件静态分派禁止降级)
12. [禁止 JIT 静默降级](#-铁律-no-silent-fallback--禁止-jit-codegen-静默降级)
13. [禁止 Scalar 调用](#-铁律-no-scalar--scalar-函数全面禁止)
14. [禁止 Fallback 绕过](#-铁律-no-fallback--禁止-fallback-绕过)
15. [禁止孤岛模块](#-铁律-no-island-module--禁止孤岛模块)
16. [禁止外部推理引擎](#-铁律-req-arch-003--禁止引入推理引擎算法库依赖)
17. [禁止外部 BLAS](#-铁律-gpu-gemm-全-jit--禁止调用外部-blas-库)
18. [禁止误导术语](#-铁律-arch-no-misleading-terminology--反误导术语)
19. [Gather 走 JIT](#-铁律-arch-gather-jit--gather-必须走-jit-管线)
20. [环境变量](#-铁律-禁止擅自添加环境变量)
21. [并发 Agent 上限](#-铁律-agent-concurrency-limit--并发子-agent-上限-6)
22. [治本不治标](#-铁律-arch-root-cause--治本不治标永不接受简单方案)

---

## 🚨 铁律 ARCH-JIT-DATA-YIELDS — JIT 代码顺从数据，而非数据顺从代码

**JIT 编译器的根本思维差异：普通软件 = 代码不变，数据适配代码；JIT = 数据不变，代码适配数据。**

JIT 编译器在编译时看到什么数据（权重 dtype/布局/shape、KV cache 结构、运行时参数），就生成与之匹配的机器码。禁止为了复用代码而改变数据布局、添加运行时分支、或要求调用方预处理数据。

### 核心原则

1. **代码生成 = 数据的纯函数**：相同的输入数据 → 相同的机器码；数据变化 → 代码变化
2. **数据是 immutable 的**：JIT 代码必须顺从实际的数据布局/dtype/shape，禁止要求调用方转换数据来迁就代码
3. **编译时特化，运行时零适配**：数据差异在编译时全部 bake 进机器码，运行时零 dtype 分支、零布局分支、零 shape 分支
4. **遇到什么 op 就生成什么代码**：编译器不前置扫描图做 bool 判断再走不同路径，而是 op lowering 按需生成，零前置假设

### 禁止规则（数据顺从代码的典型违规）

- ❌ 禁止 `if has_argmax { path_A } else { path_B }`：编译器前置 bool 门控不同代码路径
- ❌ 禁止 `LazyAbiLoad::not_needed()`：prologue 阶段前置判断"图不需要此指针"
- ❌ 禁止 `if graph.ops.iter().any(...)` 在 prologue 中决定 ABI 指针是否加载
- ❌ 禁止运行时 `match dtype` 分支：dtype 在编译时 bake 进机器码
- ❌ 禁止为了代码路径统一而要求调用方转换数据布局（如要求 BF16 权重转为 F32 再传入）
- ❌ 禁止为了复用同一段 JIT 代码而限制输入范围（如"只支持 F32 计算"）
- ❌ 禁止 prologue 阶段的 `has_xxx`/`is_xxx`/`needs_xxx` bool 变量控制代码生成

### 正确模式（代码顺从数据）

- ✅ `match topology.output_sink { ScratchpadLogits => ..., CallerBuffer => ... }`：拓扑枚举驱动
- ✅ `LazyAbiLoad::pending()` + op lowering 按需 `demand()`：遇到什么 op 就加载什么
- ✅ `dtype.x86_elem_strategy()` → 编译时特化：不同 dtype 生成不同机器码
- ✅ `SymDim::Symbolic` → 运行时从参数读值：代码不假设固定维度
- ✅ `DeviceProfile` 驱动指令选择：不同硬件生成不同指令序列
- ✅ `GraphTopologyAnalysis` 从图 ops 推导拓扑枚举：不是 bool flag

### 违规判定标准

| 模式 | 判定 |
|------|------|
| `let has_xxx = graph.ops.iter().any(...)` 在 prologue | ❌ 前置 bool flag |
| `if has_xxx { emit_A() } else { emit_B() }` | ❌ bool 门控代码生成 |
| `abi.xxx.demand().ok_or_else(...)` | ❌ Option demand = NotNeeded 状态换皮 |
| `match topology.output_sink { ... }` | ✅ 拓扑枚举驱动 |
| `abi.xxx.demand(prog, sym_map)` 返回 VRegId | ✅ 按需加载，demand 在确认需要的分支内 |
| `layer_loop_counter.is_some()` | ✅ 编译状态查询（已分配/未分配），不是 bool flag |

---

## 🚨 铁律 ARCH-JIT-YIELDS — JIT 代码顺从硬件/输入/配置/参数而动

**最高宪法。JIT 生成的每一行机器码都必须顺从（yield to）四重信息源，而非反过来要求信息源适应代码。**

### 四重信息源（优先级从高到低）

| 优先级 | 信息源 | 含义 | 示例 |
|--------|--------|------|------|
| P0 | **硬件信息** | CPU ISA 特性、GPU SM 版本、SIMD 宽度、缓存拓扑 | AVX-512 → 512-bit 向量；SM 8.0 → Tensor Core；NEON 无 SVE → 128-bit |
| P1 | **输入文件** | 权重张量 dtype/layout/shape、GGUF/SafeTensors 元数据 | Q4_1 权重 → nibble 解码+FMA+min_offset；BF16 权重 → 直接 F32 计算 |
| P2 | **用户配置** | DeviceProfile、ComputeProfile、量化策略选择 | 用户选 `SimdAssisted` → Assisted GEMV；用户选 `NativeInt8` → Int8 dot-product |
| P3 | **运行时参数** | batch_size、seq_len、num_layers 等动态维度 | seq_len=1 → decode 路径；seq_len>1 → prefill 路径；num_layers → 循环次数 |

### 核心原则

1. **代码顺从数据，数据不顺从代码**：JIT 代码的指令选择、向量化策略、内存布局完全由四重信息源驱动，禁止代码中有任何硬编码的"假设"
2. **编译时特化，运行时零分支**：四重信息源在编译时全部 bake 进机器码，运行时零 dtype 分支、零硬件分支、零 layout 分支
3. **信息源变化 → 代码变化**：权重从 Q4_1 换成 Q8_0，生成的 GEMV 代码完全不同（不是同一个函数换参数）
4. **禁止削足适履**：硬件不支持某种操作 → 必须为该硬件生成替代的最优路径，而不是改变数据布局来迁就单一代码路径

### 禁止规则

- ❌ 禁止硬编码 `QuantPrecision::F32`：dtype 必须从权重 TensorMeta 推断
- ❌ 禁止硬编码 `elem_bytes = 4`：计算精度由 dtype 传播链决定
- ❌ 禁止所有模型/所有量化格式共享同一 GEMV 代码路径：必须按 dtype 生成特化代码
- ❌ 禁止运行时 `match dtype` 分支：dtype 在编译时 bake 进机器码
- ❌ 禁止硬件不支持时 fallback 到通用路径：必须为每种硬件生成最优路径
- ❌ 禁止代码假设特定的权重存储顺序（如 lo/hi 交错 vs 拆分）：代码必须顺从实际数据布局

### 性能保证

此宪法是实现最高性能的**充要条件**：
- **充分性**：每行机器码都是为当前硬件+当前权重+当前配置的最优特化 → 零浪费
- **必要性**：任何"通用路径"都包含运行时分支和多余的内存操作 → 不可能比特化路径更快
- **量化证明**：假设通用路径每条指令平均多 1 个分支判断 → N 条指令多 N 周期；特化路径零额外周期 → 性能差距随指令数线性增长

---

## 🚨 三仓约定 (TRIPLE-REPO-PAIR)

`gllm`、`../gllm-kernels`、`../gllm-nccl` 是同一工程在三个 git 仓库中的切分。

**依赖方向（单向，禁止循环）**：`gllm → gllm-kernels → gllm-nccl`

- `gllm` 依赖 `gllm-kernels`（JIT codegen + 算子）；`gllm-kernels` 依赖 `gllm-nccl`（feature-gated by `nccl`）
- `gllm` **不直接**依赖 `gllm-nccl`（通信通过 `gllm-kernels` VmInstr call stub 间接使用）
- `gllm-nccl` 自包含：零 `gllm` 生态依赖，raw pointer API（`*mut u8`）

**统一工程规则**：
- ✅ 三仓视为同一项目；编码前 git 清洁检查必须同时覆盖三仓；任一仓脏则工作区脏
- ✅ SPEC 跨仓引用标注 `../gllm-kernels/SPEC/` / `../gllm-nccl/SPEC/`
- ❌ 禁止以"另一个仓库的问题"为由绕过规则；禁止单独审计一个仓就宣告完成

---

## 🚨 铁律 ARCH-BUILD-COMPILE-BOUNDARY — BUILD vs COMPILE 边界

**编译器 = 喂什么编译什么。BUILD 阶段根据 Family/OutputMode 选择策略裁剪图，COMPILE 阶段只看图的 OpKind 存在性/参数。**

### BUILD 阶段（graph 构建时）
- ✅ 允许根据 Family/OutputMode 选择图构建策略（Strategy Pattern）
- ✅ 允许根据模型类型裁剪图尾部 ops
- ✅ 变量名**必须**使用 `_family` 后缀：`is_encoder_family`、`is_embedding_family`
- ✅ 输出模式（OutputMode）只裁剪尾部 ops，不修改模型大类

### COMPILE 阶段（JIT 编译时）
- ✅ 只看图的 OpKind 存在性/参数；通过 `GraphTopologyAnalysis` 推导一切
- ✅ 编译产出 = 输入图的纯函数（相同图 → 相同机器码）
- ❌ 禁止读取 Family/OutputMode/BusinessConfig
- ❌ 禁止 `is_encoder`/`is_decoder` 变量名（无 `_family` 后缀 = 编译器分支）
- ❌ 禁止编译器行为依赖图结构以外的任何配置

### 违规判定

| 代码位置 | 变量名 | 判定 |
|---------|--------|------|
| BUILD 阶段 | `is_encoder` | ❌ 无 _family 后缀 |
| BUILD 阶段 | `is_encoder_family` | ✅ |
| COMPILE 阶段 | `is_encoder` | ❌❌ 严重违规 |
| COMPILE 阶段 | `topology.has_generate_loop` | ✅ 拓扑推导 |

---

## 🚨 铁律 ARCH-ROOT-CAUSE — 治本不治标，永不接受简单方案

- ❌ 禁止 "暂时只支持 X，Y 后续补" — 功能要么完整实现，要么不做
- ❌ 禁止 patch 表面症状而不修复根本架构缺陷 — 根因在架构层就重构架构
- ❌ 禁止 "先能跑起来" 的降级思维 — 架构正确性 > 短期可用性
- ✅ 治本路径: 发现问题 → 追溯到架构根因 → 重构架构消除整类问题 → 验证
- ✅ 一步到位: 每次触碰一个模块，必须把该模块的架构做到位，不留技术债

---

## 🚨 铁律 AGENT-CONCURRENCY-LIMIT — 并发子 Agent 上限 6

- ❌ 禁止同时启动超过 6 个后台子 Agent；禁止一批未完成就启动下一批
- ✅ 等待当前批次全部完成后再启动下一批；单批次内 1-6 个

---

## 🚨 铁律 ARCH-RUST-IS-CODEGEN — Rust = 代码生成器，推理阶段什么都不做

**Rust 的定位**: 代码生成器 + Hook ABI 工具库。

- ❌ 禁止 Rust 参与推理/计算/数据搬运/文本解码/采样/KV cache 管理/循环
- ❌ 禁止在热路径中出现 HashMap、String、Vec、for 循环、clone、malloc
- ❌ 禁止任何形式的 Rust 编排引擎（逐节点/逐层循环）
- ✅ Rust 只做: 模型加载时生成 JIT 机器码 → 推理时调用一次 JIT 入口函数
- ✅ 所有功能全部 JIT: forward、采样、generate 循环、KV cache、stop condition、token→text、Guardrail、SG、HR、Intent、CoT、Early Exit、MoE — 全部编译为机器码
- ✅ Hook: JIT 内嵌条件 JMP。无 hook = 不生成跳转代码。Hook 通信 = 共享内存，不经过 Rust

**推理时 Rust 的全部操作**: 一次 CALL。返回后 output_buffer 中已有完整 UTF-8 文本。

---

## 🚨 铁律 ARCH-UNIFIED-COMPILER — 统一编译器，不假设图结构

> **SSOT**: SPEC 39 + SPEC 40

**核心原则**：编译器 = 喂什么编译什么，不假设图结构。

- ✅ 唯一编译入口 `compile()`，所有图走同一管线产出 `MegaKernelFn` ABI
- ✅ `GraphTopologyAnalysis` 从图拓扑（OpKind 存在性）推导一切，替代 bool 参数
- ✅ Phase 3 按融合组顺序 emit，不硬编码执行阶段
- ✅ `GroupMarker` 机制替代硬编码层循环
- ✅ `PhaseDispatch` 仅多步生成图存在
- ✅ JIT 缓存粒度 = `MegaKernelFn`（单次 `compile()` 产出），不是全层融合图
- ❌ `CompiledLayerFn`/10-param ABI/`compile_graph` — 已物理删除，禁止恢复
- ❌ `is_encoder`/`needs_kv_for_decode` — 已删除，禁止恢复

**编译管线四阶段** (`gllm-kernels/src/compiler/`):

| 阶段 | 模块 | 职责 |
|------|------|------|
| Phase 0: Scalar + SymExec | `registry.rs` + `symexec/` | 标量参考实现 → 符号执行提取 `OpTrace` + `ComputePattern` |
| Phase 1: SemanticDAG | `semantic_dag.rs` | CompilerGraph + OpTrace → OpClass 自动分类 |
| Phase 2: Fusion + HW | `fusion.rs` + `hw_constraints.rs` | 融合决策 + 缓存层级约束 |
| Phase 3: ISA Lowering | `codegen/x86_64.rs` / `aarch64_dynasm.rs` / `gpu_ir/` | DeviceProfile 驱动代码生成 |

**融合模式** (gllm-kernels 层): EpilogueInjection / LoopFusion / TileLevelFusion / ComputeRoot / QkvSharedInput / NormIntoGemm

**融合模式** (gllm 图优化层, `src/graph/`): FlashAttention / GQA / FusedQkvRope / SwiGLU / MoERouting / FusedRMSLinear / FusedQkvNormRope (Gemma 4 六算子融合)

**设备适配** (`DeviceProfile`):
- x86_64: SSE2 → AVX → AVX2 → AVX-512（参数化 `simd_width` / `use_avx512`）
- AArch64: NEON → SVE（参数化 `use_neon` / `use_sve`）
- GPU: CUDA (PTX) / HIP (AMDGPU) / Metal (AIR)（feature-gated）
- PTX 多版本调度: `PtxKernelRegistry` 按 SM 版本选择最优内核，禁止 Fallback

**JIT 缓存协议**：
- 编译只发生在模型加载时，推理热路径禁止任何编译行为
- 动态维度通过 `SymDim::Symbolic` + `ShapeBinding` 运行时绑定，不触发重编译
- 详见 `SPEC/DOCS/scheduling/jit-cache-protocol.html`

**CPU/GPU 统一 (ARCH-CPU-GPU-UNIFIED)**：
- CPU 和 GPU 共享完全一致的 `CompilerGraph` IR 和 `GraphType`，通过 `DeviceProfile` 驱动不同 codegen
- 禁止以 `Cpu`/`Gpu`/后端名为前缀的 `GraphType` 变体
- 禁止 CPU 专用图构建器；禁止子算子级 `GraphType`

---

## 🚨 铁律 ARCH-AUTO-INSTR-SELECT — 自动指令选择器

**正确架构（类似 LLVM SelectionDAG）**：
```
Scalar → SymExec → TraceOp → [自动指令选择] → VmInstr → ISA Lowering → Machine Code
                                    ↑
                               ComputePattern 驱动
                               算法保证正确性
```

**铁律**：

1. ❌ 禁止手写 TraceOp → VmInstr 映射：必须用 `auto_lower_trace()` 查表法
2. ❌ 禁止手写 OpKind → VmInstr match arm：必须基于 `ComputePattern` 自动分发
3. ❌ 禁止 opaque 算子跳过 trace：所有 OpKind 必须在 `ScalarOpRegistry` 注册
4. ❌ 禁止创建 per-OpKind 手写函数（如 `emit_meanpool_auto`/`lower_layernorm`）：每种 ComputePattern 只有一个通用处理器
5. ❌ 禁止以"结构型算子"为由绕过 auto_select：Gather/Attention/MoE 等必须通过 TraceOp 语义扩展纳入
6. ✅ TraceOp 语义扩展是第一选择：新增变体只需在 `auto_select.rs` 添加 match arm
7. ✅ ComputePattern 通用处理器（7 种）：Elementwise / BinaryElementwise / Injective / Reduction / NormLike / Gemm / QuantDecode 各一个
8. ✅ 新增算子只需：注册 scalar impl → SymExec 提取 ComputePattern → 自动路由

**TraceOp 语义扩展原则**：
- 语义不足 = TraceOp 需要扩展，这是正常架构演进
- 新增流程：(1) `trace.rs` 添加枚举变体 (2) `auto_select.rs` 添加 match arm (3) `verify.rs` 确保 def-before-use
- 禁止以"后端约束"为由拒绝扩展 TraceOp：后端差异由 ISA Lowering 处理

**实现状态**：Phase 1-5 全部完成。`auto_lower_trace()` / ComputePattern dispatch / TraceOp 扩展 / 结构算子 TraceOp 化 / 手写 lowering 清除。

---

## 🚨 铁律 JIT 编译管线

**所有算子必须走完整 JIT 编译管线，无例外**：

```
算法 (Scalar Rust) → Lifting (SymExec trace) → IR (TraceOp SSA) → ISA Lowering (DeviceProfile) → 机器码
```

- ❌ 禁止跳过 Lifting 直接手写 ISA 汇编
- ❌ 禁止在 JIT 代码中 `call` 预编译的 Rust/C 函数
- ❌ 禁止把 JIT 当成"调度器"——只串联预编译函数
- ❌ 禁止写死特定 ISA，必须通过 `DeviceProfile` 参数化
- ✅ 新算子必须：注册 scalar 参考实现 → SymExec 提取 trace → codegen 根据硬件 lower
- ✅ 每个算子的 scalar 参考实现是 ground truth，JIT 代码必须与之数值一致

推论：宁可暂时不实现某个算子（让测试失败），也不绕过管线。`ScalarOpRegistry` 是算子注册唯一入口。

---

## 🚨 铁律 ARCH-DTYPE-JIT-TYPED — 编译时 dtype 感知

> **SSOT**: `../gllm-kernels/SPEC/GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.html §0.8` (REQ-DTYPE-001~008)

**dtype 从模型 TensorMeta 自动推断，JIT 编译时生成特化机器码，运行时零分支。**

### dtype 传播链（唯一正确路径）

```
TensorMeta.dtype (DType)
  → .to_quant_precision() (QuantPrecision)
    → op_input_dtype(op, graph) [plan_lower.rs]
      → auto_lower_trace(prog, body, inputs, width, dtype) [auto_select.rs]
        → VmInstr { ..., dtype } [instr.rs]
          → dtype.x86_elem_strategy() / .aarch64_elem_strategy()
            → Native(elem_bytes) / WidenCompute / Unsupported
```

**传播方向严格单向：tensor metadata → VmInstr 构造 → ISA lowering。禁止反向推断或独立计算。**

### 唯一合法的 `QuantPrecision::F32` 出现场景

1. ✅ 无输入 tensor 的安全回退：`op_input_dtype()` 的 `.unwrap_or(QuantPrecision::F32)`
2. ✅ QTapSTG 验证后使用：`lower_qtap_stg()` 内部已验证 dtype != F32
3. ✅ 测试代码 fixture
4. ✅ `infer_result_dtype()` 的零 slot 回退（不可能发生的边界条件）
5. ❌ 除此之外任何 `QuantPrecision::F32` 的出现都是 bug

### 禁止规则

1. ❌ 禁止硬编码 `QuantPrecision::F32`：dtype 必须从 `graph.tensor(op.inputs[0]).dtype` 推断
2. ❌ 禁止 `computation_elem_bytes()` 硬编码 F32：必须用 `op_input_dtype(op, graph).elem_bytes()`
3. ❌ 禁止硬件不支持时 fallback 到 F32：必须返回 Error
4. ❌ 禁止运行时 `match dtype` 分支：dtype 在编译时 bake 进机器码
5. ❌ 禁止 dtype 身份匹配驱动指令选择：使用 `dtype.x86_elem_strategy()` 属性方法
6. ❌ 禁止在 VmInstr 构造时省略 dtype 传播

---

## 🚨 铁律 ARCH-SYMDIM-NO-CONST-DEGRADE — SymDim 穿透禁止降级为编译时常量

**`SymDim::Symbolic` 维度在整个管线中必须保持 Symbolic 语义，禁止降级为编译时常量。**

1. ❌ 禁止 `max_for_allocation()` 作为运算参数：仅用于 buffer 分配大小计算
2. ❌ 禁止 `BoundExpr::Const` 处理 Symbolic 维度：必须用 `BoundExpr::Symbolic` 从运行时参数读取
3. ❌ 禁止 `infer_elem_count` 返回固定值：必须返回 Symbolic 信息
4. ❌ 禁止 executor 层手动计算 seq_len：必须从图的 SymDim 元数据正向传递
5. ❌ 禁止编译时固定 `compile_seq_len`
6. ❌ 禁止 `SYMDIM_MAX_SEQ_LEN` 作为运算参数：仅用于 buffer 分配上界

正确架构：`SymDim::Symbolic("seq_len") → BoundExpr::Symbolic → JIT 循环从 [rbp+16] 读运行时值`

---

## 🚨 铁律 NO-LAYER-EXPAND — 禁止逐层展开

**层模板编译一次，运行时循环 N 次。禁止为每层单独生成 op。**

- ❌ `for i in 0..num_layers { emit_op(&format!("L{i}_k_proj")) }` — 逐层展开
- ❌ 构建 N 份相同或相似模板；VmInstr 数量与 num_layers 成正比
- ✅ 单模板 + `GprCondAction` 运行时条件分支
- ✅ `build_compiler_graph` 构建单模板（"layer." 前缀），所有层共享同一套 op
- ✅ `num_layers` 不同 → 同一个 JIT 产物，只是循环次数不同

**SharedKvRef 条件分支（Gemma 4）**：条件 `layer_idx < (num_layers - num_kv_shared_layers)`，Consumer 层 `GprCondAction` 跳过 K/V 相关 5 个 op。非 SharedKvRef 模型条件永远 true，零开销。

---

## 🚨 铁律 ARCH-NO-LOOP-UNROLL — 禁止 Rust 循环展开 VmInstr

**JIT codegen 中所有数据相关的循环必须使用 `emit_loop()`，禁止用 Rust `for` 循环展开为扁平 VmInstr 序列。**

- ❌ 禁止 `for qi in 0..seq_len` / `for h in 0..num_heads` 展开 VmInstr
- ✅ 必须使用 `prog.emit_loop(BoundExpr, step, |prog, ctr, off| { ... })`
- ✅ `BoundExpr::Const` 仅用于编译时确定且极小的内层微维度（head_dim/lanes, ≤ UNROLL_THRESHOLD）

---

## 🚨 铁律 NO-HW-DEGRADATION — 硬件静态分派禁止降级

**JIT codegen 必须为当前硬件生成最优路径代码，禁止任何形式的硬件降级/退化。**

- ❌ `HardwareFusionPass` 将融合算子降级为原子算子
- ❌ "硬件能力不足所以降级" 作为拆分融合算子的理由
- ✅ CPU 没有 FlashAttention 硬件 → codegen 生成 cache-friendly tiled attention（仍然融合）
- ✅ 硬件差异体现在 **codegen 层的指令选择**，不是 **fusion 层的算子拆分**

架构原则：`FusionRule 生成融合图 → JIT codegen 根据 DeviceProfile 为每种硬件生成最优机器码`

---

## 🚨 铁律 NO-SILENT-FALLBACK — 禁止 JIT Codegen 静默降级

**JIT codegen 遇到无法生成代码的 OpKind 必须返回 `Err`，禁止静默 NOP。**

- ❌ `emit_nop_raw()` / `match _ => Ok(())` / `eprintln!("[WARN]...")` + scalar 替代
- ✅ 未实现的 op 必须 `Err(format!("codegen not implemented for {:?}", op_kind))`
- ✅ 仅 `Reshape`/`Transpose`（纯元数据 op）允许 NOP

---

## 🚨 铁律 NO-SCALAR — Scalar 函数全面禁止

**`scalar_ops.rs` 中的函数禁止在任何代码路径中被调用（包括运行时和测试），仅供 `ScalarOpRegistry` 注册 + SymExec trace 提取。**

禁止的函数：`scalar_gemm`/`scalar_rms_norm`/`scalar_rope`/`scalar_moe_gate`/`scalar_top_k_experts`/`scalar_expert_ffn`/`scalar_moe_ffn`/`cached_gqa_attention`/`prefill_gqa_attention`/`swiglu_ffn`

- ❌ 禁止在运行时/测试/`#[cfg(test)]` 中调用任何 `scalar_*` 函数
- ❌ 禁止 `use.*scalar_ops` 出现在 `scalar_ops.rs` 以外
- ✅ 测试必须通过 JIT 编译图（`CompilerGraph` → `compile_and_run`）验证

---

## 🚨 铁律 NO-FALLBACK — 禁止 Fallback 绕过

**功能缺失必须补全实现，禁止用 fallback 绕过。**

- ❌ JIT 编译失败时 fallback 到手写 Rust；融合算子失败时降级到原子算子拼接
- ❌ 用 stub/dummy 返回值绕过未实现的功能
- ✅ 测试失败 = 功能缺失，必须补全真正的实现

**SPEC 已授权的 Fallback（仅以下 5 个）**：

| # | Fallback | SPEC 授权 |
|---|----------|----------|
| A2 | HF→ModelScope 下载源切换 | REQ-LOADER-016 |
| A3 | ONNX Fusion→Atomic（模式不匹配时） | ARCH-ONNX |
| A4 | HW Fusion→Standalone（硬件约束违反时） | ARCH-DETAILED-DESIGNS |
| A5 | Reshape/Transpose 元数据 NOP | NO_SILENT_FALLBACK 例外 |

---

## 🚨 铁律 NO-ISLAND-MODULE — 禁止孤岛模块

**新增模块必须验证真实调用链接入，禁止"编译通过+测试通过=完成"。**

1. SPEC 必须包含 Integration Trace：完整调用链标明 `文件:函数:行号`
2. 实现完成后必须 grep 验证非测试调用
3. 集成测试必须证明"不同输入 → 不同输出"通过真实路径传导

- ❌ 禁止以"编译通过+单元测试通过"为完成标准
- ❌ 禁止依赖 `Default::default()` 掩盖"没接入"
- ✅ 核心函数必须在非测试推理/加载路径上有调用点

---

## 🚨 铁律 REQ-ARCH-003 — 禁止引入推理引擎/算法库依赖

- ❌ 禁止 `candle`/`tch`/`torch`/`ort`/`tract`/`burn` 等推理框架（含 optional feature）
- ✅ 计算核心完全由 `gllm-kernels` 提供
- ✅ 仅允许底层工具库：`safetensors`/`zip`/`prost`/`half` 等

---

## 🚨 铁律 GPU GEMM 全 JIT — 禁止调用外部 BLAS 库

- ❌ 禁止在 GPU 推理路径中调用 cuBLAS/rocBLAS/cuDNN
- ✅ GPU GEMM 由 JIT 编译管线生成 PTX/HIP/MSL 原生二进制
- ✅ `GpuIsvCapabilities` 仅包含 `tensor_cores_gen`；`GemmStrategy` GPU 路径为 `JitGpuTensorCore`/`JitGpu`

---

## 🚨 铁律 ARCH-NO-MISLEADING-TERMINOLOGY — 反误导术语

**禁止使用暗示编译器区分 encoder/decoder 的术语。编译器从图拓扑推导一切，不按模型类型分支。**

| ❌ 禁止术语 | ✅ 替换术语 | 原因 |
|------------|------------|------|
| "encoder 图"/"encoder model" | "无 Argmax 的图" | 编译器按图拓扑推导，不按模型类型分支 |
| "decoder 图"/"decoder model" | "含 Argmax 的图" | 同上 |
| "encoder path"/"decoder path" | "单遍路径"/"生成循环路径" | 路径由拓扑决定 |
| "is_encoder" | `GraphTopologyAnalysis.has_generate_loop` 的否定 | bool 参数已删除 |
| "needs_kv_for_decode" | `GraphTopologyAnalysis.needs_persistent_kv_cache` | 同上 |
| "Phase N (N≥4) 作为独立执行阶段编号" | "采样管线"/"ForwardPhaseDispatch"/具体 VmInstr 名 | Phase 编号暗示硬编码阶段 |
| "encoder 输出逻辑" | "pool/classify 输出逻辑" | 描述具体操作 |
| "N 个 decoder 层" | "N 个同构子结构" | 编译器不假设层类型 |

**例外说明**：`ForwardPhaseDispatch` / `GroupMarker::PhaseDispatch` 作为 SPEC 39 REQ-UMK-012/013 定义的功能组件名称不在禁止范围。禁止对象是：以 Phase 编号隐含固定执行顺序的设计。功能组件命名（含 Phase 词根但语义为具体功能）合法。

**AI 行为约束**：
- ❌ 禁止因注释中出现"encoder"/"decoder"而假设编译器有双路径
- ❌ 禁止将 `is_encoder`/`needs_kv_for_decode` 的历史存在解释为"编译器需要区分"
- ✅ 必须从图拓扑（OpKind 存在性）推导编译器行为
- ✅ 必须将 SPEC 39 §0.1 作为编译器行为的唯一权威来源

---

## 🚨 铁律 ARCH-GATHER-JIT — Gather 必须走 JIT 管线

- ❌ 禁止 `if op.name() == "Gather" { execute_gather_cpu(...); continue; }`
- ❌ 禁止在 executor 层对特定 OpKind 做特殊分支处理
- ✅ Gather lower 必须生成正确的 VM 指令；VM 指令集不支持时先扩展再实现

### 模型配置参数禁止硬编码

- ❌ 禁止 `(i + 2) as f32` 式 position offset；禁止 `vec![0.0; seq_len]` 式 token_type_ids
- ✅ 所有模型特定参数从 `ModelConfig`/`ModelManifest` 读取

---

## 🚨 铁律 禁止擅自添加环境变量

- ❌ AI 禁止擅自引入新环境变量或为"灵活性"添加环境变量
- ✅ 新增环境变量必须由用户明确要求，且有不可替代的使用场景

---

## Core Architecture

### 0. Core Philosophy: Accuracy First

1. **Accuracy > Throughput**: TurboQuant 2.0 通过数学级静态湮灭保证量化精度
2. **Reliability First**: PagedAttention 严格边界检查，OOM rejection > corrupted results；JIT emit hardware Trap on OOB
3. **JIT Hot-Repair & Block Routing**: 运行时 JIT 原子擦写指令 + `SystemTopology` 物理块式异构动态图
4. **NO PRAGMATIC HACKS**: 禁止 `unwrap_or(default)` 等 fallback、禁止硬编码魔法数字、禁止临时 workaround

### 1. Supported Model Architectures

| 架构 | CPU JIT | GPU codegen | 备注 |
|------|--------|-------------|------|
| Qwen3 / Qwen3MoE | ✅ | ✅ | 通用标杆 |
| Llama4 / SmolLM / InternLM | ✅ | ✅ | G-A 路径共用 |
| GLM-4 / GLM-5 | ✅ | ✅ | MoE (glm-4.7-flash) |
| Mistral3 / Ministral | ✅ | ✅ | Sliding Window |
| Phi4 / Phi4-mini | ✅ | ✅ | Partial RoPE |
| GptOss (gpt-oss-20b) | ✅ | ✅ | MoE + sliding/full 交替 + yarn RoPE |
| DeepSeek V3/R1 / Kimi-K2 | ✅ | ✅ | MoE + MLA 矩阵吸收 + MTP |
| XLM-R / XLM-R-Next | ✅ | ✅ | Embedding / Rerank (无 Argmax 图) |
| **Gemma 4** | ✅ | ✅ | QkNorm + ValueNorm + DualRoPE + AltUp+PLE 🔄 + SharedKvRef + Vision ✅ + Audio ✅ |

**Gemma 4 关键差异点**：DualRoPE (sliding θ=10K+partial=1.0 / global θ=1M+partial=0.25) / QkNorm+ValueNorm / AltUp+PLE (E2B/E4B) / SharedKvRef (GprCondAction 条件跳过) / FusedQkvNormRope (6 算子融合)

---

## Directory Structure

```
src/
├── lib.rs, client.rs, model_config.rs, tokenizer.rs, generation.rs
├── embeddings.rs, rerank.rs, classify.rs, ffi.rs, routing.rs
├── quantization.rs, fp8.rs, static_compression.rs, weight_loader.rs, weight_names.rs
├── rag.rs, head_routing.rs, intent.rs, intentTracker.rs
├── cot_reasoner.rs, guardrail.rs, early_exit.rs, prefetch.rs
│
├── client_fragments/        # Client impl fragments (.inc.rs)
├── model_config_fragments/  # ModelConfig impl fragments (.inc.rs)
├── compat/                  # gllm-kernels compatibility shim
│   ├── mod.rs, types.rs, scalar_ops.rs, jit_helpers.rs, weight_helpers.rs
│   ├── cpu_backend.rs, cuda_backend.rs, hip_backend.rs, metal_backend.rs
│   ├── gpu_compile.rs, gpu_helpers.rs, gpu_backend_macro.rs, memory.rs
│   ├── multimodal.rs, audio_forward.rs, vision_forward.rs, sampling.rs
│
├── loader/                  # Model fetching & parsing
│   ├── mod.rs, adapter.rs, downloader.rs, format_detector.rs
│   ├── hf_hub.rs, modelscope.rs, safetensors.rs, pytorch.rs, parallel.rs
│   ├── name_map.rs, weight_compress.rs, weight_tier.rs
│   ├── awq_gptq_pairing.rs, mxfp4_pairing.rs, nvfp4_pairing.rs
│
├── arch/                    # Architecture auto-detection (tensor-name-driven)
│   ├── mod.rs, registry.rs, auto_graph.rs, resolve.rs, intent_tracker_graph.rs
│
├── graph/                   # DAG optimizer
│   ├── mod.rs, types.rs, layer_callback.rs, profile.rs
│
├── engine/                  # Execution engine
│   ├── mod.rs, executor.rs, executor_api.rs, executor_builder.rs
│   ├── executor_compile.rs, executor_step.rs, executor_types.rs
│   ├── batch_context.rs, batch_executor.rs, mtp_executor.rs, arbiter.rs
│   ├── mega_kernel.rs, mega_kernel_callback.rs, mega_kernel_v2.rs
│   ├── callbacks/           # Hook callbacks
│   │   ├── mod.rs, early_exit.rs, gate_skip.rs, mid_layer_encode.rs
│   │   ├── moe_dispatch.rs, rag_inject.rs, residual_bus_bridge.rs
│   ├── coordinator/         # Executor decomposition coordinators
│   │   ├── mod.rs, dispatch.rs, kv.rs, compute.rs, inference.rs
│   │   ├── callback_slot.rs, model_context.rs, observability.rs, sg_callback_handle.rs
│   ├── mega_kernel/         # Mega-kernel .inc.rs fragments
│       ├── abi_types.inc.rs, executor_core.inc.rs, executor_ops.inc.rs, pack_observe.inc.rs
│
├── jit/                     # JIT compilation integration
├── kv_cache/                # KV Cache structures
├── moe/                     # MoE dispatch & routing
├── speculative/             # Speculative decoding
├── semantic_gatekeeper/     # Semantic Gatekeeper integration
├── sensors/                 # Hardware sensors
├── scheduler/               # Batching & KV Cache management (HGAL)
│   ├── mod.rs, paged_scheduler.rs, batcher.rs, hgal.rs, allocator.rs
│   ├── memory_manager.rs, prefix_index.rs, sequence.rs, types.rs
│   ├── observer.rs, policy.rs, jit_types.rs, vllm2024.rs
│   ├── kv_optimizer.rs, chunked_prefill.rs, request_state.rs
│   ├── three_tier_swap.rs, nvme_swap.rs, dma_helpers.rs
│   ├── migration_actor.rs, eviction_worker.rs, swap_in_worker.rs
│   ├── compact.rs, fault_recovery.rs, telemetry.rs
│
├── backend/                 # Backend detection (ARCH-ZERO-FALLBACK)
│   ├── mod.rs, detection.rs
│
└── manifest/                # Model manifest types
    ├── mod.rs, types.rs
```

---

## Technology Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Loader** | `hf-hub`, `safetensors`, `prost`, `memmap2` | Model fetching, zero-copy loading, ONNX/GGUF/PyTorch parsing |
| **Tokenizer** | `tokenizers` | Text <-> ID conversion |
| **Scheduler** | Custom (HGAL) | PagedAttention, Continuous Batching, KvPrefixIndex |
| **Engine** | `gllm-kernels` + `compat/` shim | Hardware abstraction (compat bridges types + forward passes) |
| **Graph** | Custom DAG + `serde_yaml` | Unified OnnxGraph representation & fusion optimization |
| **Backend** | `gllm-kernels` | Auto-detect CUDA/ROCm/Metal/CPU, JIT codegen |
| **Distributed** | `gllm-nccl` (feature-gated `nccl`) | Multi-vendor GPU/NPU collective communication |

---

## nccl Feature — 分布式通信集成

```toml
[features]
nccl = ["gllm-kernels/nccl"]
```

- `nccl` feature 转发到 `gllm-kernels`，gllm 不直接依赖 `gllm-nccl`
- 数据流：`gllm Client → CompilerGraph (VmInstr: AllReduceChunk) → JIT call stub → gllm-nccl CommHandle`

---

## Cache Directory

- **Model Cache Root**: `~/.gllm/models/`（`GLLM_CACHE_DIR` 可覆盖）
- **JIT Cache**: `~/.gllm/jit_cache/`（L3 磁盘层，7 天 TTL）
  - Debug 模式：L3 禁用；Release 模式：正常工作

---

## Common Commands

```bash
# 编译检查
cargo check

# 单元测试（可并行）
cargo test --lib

# E2E 测试（必须单线程 + 单实例）
cargo test --test test_e2e_embedding -- --test-threads=1
cargo test --test test_e2e_generator -- --test-threads=1
cargo test --test test_e2e_reranker -- --test-threads=1

# gllm-kernels 测试
cd ../gllm-kernels && cargo test --lib
cd ../gllm-kernels && cargo test --test decision_audit
```

---

## 🧪 E2E 测试约束

**单线程 + 单实例强制要求 (ARCH-SINGLE-MODEL-INSTANCE)**：
- ❌ E2E 测试禁止并行（`--test-threads=1`）；禁止同时加载超过 1 个模型实例
- ✅ 同一时刻只能有一个模型加载在内存中

**Pipeline 测试原则**：直接用现有 API 跑本地模型，哪里断了就是真实 bug，不做 workaround。

---

## CompilerGraph OpKind (Gemma 4 新增算子)

- `QkNorm { head_dim }`: Q/K 向量 L2 归一化 + √head_dim 缩放
- `ValueNorm { eps }`: V 向量无学习参数 RMSNorm
- `AltUpPredict { num_preds, hidden }`: AltUp 预测路混合
- `AltUpCorrect { num_preds, hidden }`: AltUp 修正路缩放
- `AltUpInject { num_preds, hidden }`: PLE 门控注入
- `ColumnSlice { start, end }`: 按层切片 JIT 实现
- `PatchEmbed` (Vision): 图像 → patch 序列
- `LearnedPos2D` (Vision): 2D 学习位置编码
- `DepthwiseConv1D` (Audio): Conformer 深度卷积
- `RoPE { partial: f32 }` (扩展): DualRoPE 支持 partial p-RoPE

---

## 技能索引

| 技能 | 描述 |
|------|------|
| `/root-cause-debugging-philosophy` | 高维度根本性根治调试哲学与SPEC驱动开发 |
| `/concurrent-agent-orchestration` | 并发子agent编排：波次调度、冲突避让、用量限额恢复策略 |
| `/e2e-test-debugging-workflow` | E2E测试调试与数值对齐工作流 |
| `/jit-pipeline-development` | JIT编译管线开发规范与算子补充流程 |

---

## 分布式测试环境

### 本地机器 (pt-worker)

| 项目 | 值 |
|------|-----|
| IP | 192.168.1.205 |
| OS | Linux 6.17 (x86_64) |
| CPU | Intel i9-10900KF (10C/20T) |
| RAM | 128 GB |
| GPU | NVIDIA GTX 1060 6GB (SM 6.1) |
| 角色 | CPU E2E 测试 + 编译验证 + 调度 |

### 5070 Ti 服务器 (WSL2)

| 项目 | 值 |
|------|-----|
| IP | 192.168.1.200 |
| OS | WSL2 Linux 6.6.114 (x86_64) on Windows |
| CPU | AMD Ryzen 9 9950X3D (16C/32T) |
| RAM | 24 GB |
| GPU | NVIDIA RTX 5070 Ti 16GB (SM 12.0, Blackwell) |
| SSH | `sshpass -p '123456' ssh -o StrictHostKeyChecking=no putao@192.168.1.200` |
| nvidia-smi | `/usr/lib/wsl/lib/nvidia-smi` |
| 角色 | GPU E2E 测试 + CUDA codegen 验证 |

### 分布式 E2E 测试

- 单机多卡：需 2+ GPU 机器（当前无可用的）
- 多机多卡：需 2 台机器 + NCCL 运行时（本地 1060 不支持 NCCL，5070 Ti 单卡）
- gllm-nccl `cargo test --lib` 需 `libnccl.so.2`
- `cargo check --features nccl` 编译通过 ✅
