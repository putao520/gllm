# 计算图中间表示

> **SSOT**: 本文档是 gllm + gllm-kernels 统一项目计算图 IR（中间表示）的唯一真源。定义 CompilerGraph SSA IR、OpKind 算子语义、SymDim 动态维度系统。
>
> **架构 (REQ-UGS-005)**: FusedGraph/FusedOp/FusedNode/FusedGraphExecutor 及 graph/optimizer/ 整个目录已物理删除。Mega-Kernel 是系统唯一的编译和执行路径。`auto_graph` 直接构建 `CompilerGraph`，无中间表示层。
>
> 交叉引用: `01-JIT-PIPELINE.md`（JIT 管线、auto_select 指令选择、TraceOp→VmInstr 映射）、`02-HARDWARE.md`（DeviceProfile）、`04-OPERATORS.md`（算子库与硬件特化）

## 1. CompilerGraph — 唯一图表示

系统通过 CompilerGraph 连接 auto_graph 输入到 Mega-Kernel JIT 编译器：

```
auto_graph (tensor-name-driven) → CompilerGraph → compile() → 单次 CALL → 推理结果
     (架构推导)                   (JIT 内部 SSA IR)     (全模型一次编译)
```

CompilerGraph 是 JIT 四阶段管线的核心数据结构。张量通过 `SymDim` 表达形状，算子通过 `OpKind` 表达语义。

```rust
pub struct CompilerGraph {
    pub ops: Vec<CompilerOp>,
    pub tensors: HashMap<TensorId, TensorDesc>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

/// 张量描述 — 携带量化元数据供 auto_select 使用 (ARCH-COMPUTE-TIME-DEQUANT)
pub struct TensorDesc {
    pub shape: Vec<SymDim>,
    pub dtype: DType,                // 计算精度 (从模型元数据推断)
    pub source_quant: QuantType,     // 原始量化格式 (Q4_K / BF16 / F32 原生 / MXFP4)
    pub block_size: Option<usize>,   // 量化块大小 (MXFP4=32, Q4_K=256, F32=None)
}
```

### 1.1 量化元数据流 (ARCH-COMPUTE-TIME-DEQUANT)

```
Loader: GGUF Q4_K → QuantType::Q4_K { block_size: 256 }
    ↓
gllm: auto_graph → TensorDesc { dtype: compute_dtype, source_quant: Q4_K, block_size: Some(256) }
    ↑ dtype 由 (source_quant, DeviceProfile) 推导, 非 F32 硬编码
    ↓
gllm-kernels: auto_select 读 TensorDesc.source_quant
    → 匹配 DequantComputeVariant::FusedQ4Gemm { block_size: 256 }
    ↓
Phase 3: 生成融合 "读 4bit → 解包 → compute_dtype 乘累加" 代码
```

| source_quant | JIT 行为 |
|-------------|---------|
| `F32` | 直接计算，无 dequant 步骤 |
| `BF16` | 硬件有 BF16 指令 (VDPBF16PS / BF16 WMMA) → 原生路径；否则 → 软件转换 |
| `Q4_K` / `Q4_0` | FusedQ4Gemm: 微核内循环解包 4bit + scale → compute_dtype 乘累加 |
| `Q8_0` / `Q8_K` | FusedQ8Gemm: 微核内循环 8bit × scale → compute_dtype 乘累加 |
| `MXFP4` | FusedMxfp4Gemm: 双指针 (blocks + scales) 读取 |
| `Q4_K × Q8_0` on GFX12 | Rdna4WmmaA4W8: 硬件隐式 dequant，零软件开销 |

### 1.2 OpKind — 唯一算子语义枚举

所有算子通过 `OpKind` 枚举表达。OpKind 定义在 `gllm-kernels/src/compiler/graph.rs`，是 JIT 管线的唯一算子入口。每个 OpKind 对应一个独立的计算语义，携带完整的配置参数。

完整 OpKind 清单见 `SPEC/04-OPERATORS.md` §1。关键 OpKind 包括：

| OpKind | 语义 | 关键参数 |
|--------|------|---------|
| `Gemm` | 矩阵乘法 | m, n, k, dtype |
| `RmsNorm` | RMS 归一化 | hidden_size, eps |
| `RoPE` | 旋转位置编码 | head_dim, theta, partial |
| `MultiHeadAttention` | 多头注意力 | num_heads, num_kv_heads, head_dim, scale |
| `Silu` / `Mul` / `Add` | 逐元素操作 | — |
| `Gather` | Embedding lookup | table_rows, embed_dim |
| `MoEGate` / `MoERouter` / `MoEDispatchPacked` | MoE 算子族 | num_experts, top_k |
| `QkNorm` / `ValueNorm` | Gemma 4 QK/V 归一化 | head_dim, eps |
| `PerLayerEmbed` / `ColumnSlice` | Gemma 4 PLE | layer_idx, dim_per_layer, start, end |
| `SgDetect` / `SgInject` | Semantic Gatekeeper 共享内存 | detect_offset, knowledge_offset |
| `SessionKvRestore` | Session KV 复用 | session_position |
| `MmHiddenInject` | 多模态 hidden 注入 | num_mm_tokens |
| `StoreToken` / `CheckStopCondition` | Generate loop 控制 | eos_token_id, max_tokens |

### 1.3 auto_graph → CompilerGraph 直接构建

`auto_graph` 根据 tensor names 推导架构特征，直接构建 CompilerGraph（REQ-UGS-001）。无中间表示层。

**构建流程**：
1. 从模型权重 tensor names 推导架构特征（hidden_size、num_heads、MoE、GQA 等）
2. `op_type` 字符串 → `OpKind` 枚举映射（Gather→OpKind::Gather, MatMul→OpKind::Gemm 等）
3. 模型配置属性 → OpKind 字段注入（theta→RoPE.theta, eps→RmsNorm.eps 等）
4. **运行时条件分支**：层模板内的部分 op 可附带条件守卫，编译为 `VmInstr::GprCondAction`。JIT 层循环中 GPR 持有当前 `layer_idx`，条件分支检查阈值决定是否跳过该 op 组（见 §1.3.1 SharedKvRef 条件分支）
5. `repeat` 块编译为 `emit_loop(BoundExpr::Const(num_layers), weight_stride)` — JIT 内部循环，单模板运行 N 次
6. Mega-Kernel 独有节点由 `MegaKernelBusinessConfig` 驱动注入

**唯一数据流**: `auto_graph` → `CompilerGraph` → `compile()` → 单次 CALL

#### §1.3.1 SharedKvRef 条件分支（Gemma 4）

Gemma 4 E2B/E4B/26B-A4B/31B 的后 `num_kv_shared_layers` 层（consumer）跳过 K/V 投影，直接引用 donor 层的 KV cache。通过运行时条件分支实现，编译为单模板，零代码膨胀。

**条件表达式**：`layer_idx < (num_layers - num_kv_shared_layers)`

| 层类型 | 条件结果 | 执行路径 |
|--------|---------|---------|
| Donor（前 N-S 层）| true | k_proj → k_norm → v_proj → v_norm → rope_k → 写入自身 KV cache → MHA |
| Consumer（后 S 层）| false | GprCondAction 跳过上述 5 个 op → MHA 直接读 donor KV cache page |

**无条件执行的 op**（所有层）：
- input_norm → q_proj → q_norm → rope_q → MHA → o_proj → resid → post_norm → FFN

**编译产物**：层模板中 consumer 独有的 5 个 op（k_proj, k_norm, v_proj, v_norm, rope_k）前插入 `GprCondAction { condition: layer_idx < threshold, action: Skip(N) }`。N = 这 5 个 op 编译产生的 VmInstr 总数。

**零开销保证**：非 SharedKvRef 模型的 `num_kv_shared_layers = 0`，threshold = num_layers，条件永远 true，Skip 永远不触发。

**Donor 匹配规则**（正确性关键约束）：

Consumer 层必须映射到**同 attention 类型**的最后一个 donor 层。Gemma 4 有 sliding/global 两种 attention 类型，RoPE theta 不同（sliding=10K, global=1M），consumer 层读 donor KV 时 donor 的 rope_k 已应用了对应 theta 的位置编码。如果 consumer 映射到不同类型的 donor，RoPE theta 不一致导致位置编码混乱（vLLM #39914 根因）。

```
donor_match(consumer_idx):
    consumer_type = attention_pattern[consumer_idx]
    donor_candidates = [0 .. num_layers - num_kv_shared_layers)
    donor_idx = donor_candidates 中最后一个 attention_pattern == consumer_type 的层
```

映射示例（Gemma 4 E2B, 14 层, 12 shared, pattern=[s,s,s,s,f, s,s,s,s,f, s,s,s,s]）：
- Layer 2 (s) → donor = Layer 0 (s) — 同类型最后一个
- Layer 3 (f) → donor = Layer 1 (f) — 同类型最后一个
- Layer 4 (s) → donor = Layer 0 (s)
- ...以此类推

**KV cache page 共享**（调度器层，已实现 T39）：
- `effective_kv_layers = num_layers - num_kv_shared_layers`
- `effective_kv_layer(layer_idx)` 使用上述 donor 匹配规则映射 consumer → donor 层索引
- Consumer 层 MHA 的 PagedAttention 间接寻址指向 donor 写入的物理页
- Consumer 层不分配独立 KV cache slot

**§1.3.2 SharedKvRef Prefill 两阶段加速**

标准 prefill 中所有 N 层顺序执行，consumer 层虽跳过 K/V GEMM 但其 MHA 读 donor KV 时 donor 可能尚未写入（同一 prefill 内层间顺序依赖）。SharedKvRef prefill 加速通过**两阶段执行**消除等待：

```
阶段 1 (Donor Pass): 执行层 [0, num_layers - num_kv_shared_layers)
  → 完整计算: Q + K + V + RoPE + Attention + FFN
  → KV cache 完整写入所有 donor 层的物理页

阶段 2 (Consumer Pass): 执行层 [num_layers - num_kv_shared_layers, num_layers)
  → 跳过 K/V GEMM (GprCondAction)
  → MHA 直接读阶段 1 已写入的 donor KV cache
  → KV cache 无写入（consumer 不拥有 KV slot）
```

**性能收益**：
- Donor pass 完成后，所有 donor KV cache 页驻留 L2 cache → consumer pass 的 MHA 读取命中 L2
- 节省 `num_kv_shared_layers × 2` 次 K/V GEMM（prefill 阶段 K/V 投影占总 FLOPs 的 ~30%）
- vLLM 等价：Self-Decoder / Cross-Decoder 分组执行
- SwiftKV 研究（arXiv 2410.03960）表明此类优化可实现 ~2x prefill 吞吐

**Mega-Kernel 集成**（SPEC 32 ForwardPhaseDispatch 扩展）：
- ForwardPhaseDispatch 检测 `num_kv_shared_layers > 0` → 启用两阶段 prefill
- 阶段 1：`layer_idx` 循环 [0, donor_count)，所有 op 无条件执行
- 阶段 2：`layer_idx` 循环 [donor_count, num_layers)，GprCondAction 跳过 K/V
- Decode 阶段不变（单层循环，GprCondAction 正常工作）
- 非 SharedKvRef 模型：两阶段退化为单阶段（consumer_count = 0），零开销

**vLLM 对照**：vLLM 使用 `is_kv_shared_layer` 布尔标志 + Python `if/else` 运行时分支。gllm 的等价机制是 GPR 条件比较 + VmInstr Skip。vLLM 的 consumer 层仍然执行 qkv_proj（浪费算力），gllm 通过 GprCondAction 完全跳过以节省 GEMM 计算。

### 1.4 JIT 编译管线

CompilerGraph 进入 JIT 四阶段管线，编译为单次可执行的机器码：

| 阶段 | 输入 | 输出 | 职责 |
|------|------|------|------|
| Phase 0: Scalar + SymExec | OpKind scalar 实现 | OpTrace + ComputePattern | 算子语义定义 + 符号执行 |
| Phase 1: SemanticDAG | CompilerGraph + OpTrace | OpClass 自动分类 | ElemWise/Injective/Reduction/Gemm/Opaque |
| Phase 2: Fusion + HW | OpClass + DeviceProfile | FusionPlan | 融合决策 + 寄存器/缓存约束 |
| Phase 3: ISA Lowering | FusionPlan + DeviceProfile | x86_64/AArch64/PTX/HIP/MSL 机器码 | 硬件特化代码生成 |

## 2. SymDim 动态维度系统

```rust
pub enum SymDim {
    Concrete(usize),
    Symbolic(String),
}
```

张量形状可以混合具体维度和符号维度：

```rust
// hidden_size 已知: SymDim::Concrete(hidden_size)
// total_seq, 运行时绑定: SymDim::Symbolic("total_seq")
graph.add_tensor("kv_cache", &[
    SymDim::Concrete(num_layers),
    SymDim::Concrete(kv_dim),
]);
```

运行时绑定 `ShapeBinding`：

```rust
pub struct ShapeBinding {
    pub bindings: HashMap<String, usize>,
}
```

**铁律**：
- 禁止在图构建时将 `total_seq`、`batch_size` 等动态维度硬编码为具体数值
- 每步 decode 递增 `total_seq` 时禁止重新编译图
- `Concrete` 维度在编译时参与代码生成优化（循环展开/tile 大小）；`Symbolic` 维度在运行时从 binding 读取，生成通用循环
- 缓存粒度 = 全模型 Mega-Kernel，编译一次，绑定 `ShapeBinding`

> **穿透协议**: SymDim 从 CompilerGraph 到 JIT 机器码的完整穿透机制见 `DOCS/scheduling/symdim-threading-protocol.md` (ARCH-SYMDIM-THREADING)。
> 核心要求：SymDim::Symbolic 信息必须穿透 plan_lower → lower → VmInstr，到 BoundExpr::Symbolic 自然转换为运行时循环 bound。禁止 `as_concrete().unwrap_or()` 丢弃符号信息。

## 3. Callback 机制 — Mega-Kernel 内嵌条件跳转

Callback（Guardrail、SG、HR、Intent、Early Exit）通过 **JIT 代码内嵌的条件跳转** 集成。没有 callback 注册时不生成任何跳转代码，机器码中不存在 callback 相关指令。

**编译时行为**：
- 无 callback 注册 → 不生成任何跳转/分支/trap 代码。层循环是连续的直线代码
- 有 callback 注册 → 生成条件 JMP 读共享内存状态字

**Callback 通信**：通过共享内存写入，JIT 机器码直接读取。不经过 Rust 函数调用。GPU 路径通过 global memory 写入。

CallbackAction 和 CallbackChain 完整定义见 `SPEC/05-OPTIMIZATIONS.md` §1。

## 4. 交叉引用

| 主题 | 位置 |
|------|------|
| JIT 四阶段管线 + auto_select 指令选择 | `SPEC/01-JIT-PIPELINE.md` |
| TraceOp→VmInstr 自动查表映射 | `SPEC/01-JIT-PIPELINE.md` §6 |
| 硬件探测与 DeviceProfile | `SPEC/02-HARDWARE.md` |
| OpKind 算子库 + 硬件特化路径 | `SPEC/04-OPERATORS.md` |
| 融合策略差异矩阵 | `SPEC/05-OPTIMIZATIONS.md` |
| Per-Node Callback 架构 | `SPEC/05-OPTIMIZATIONS.md` §1 |
| Mega-Kernel 编译 + ABI + Session/MM | `SPEC/08-EXECUTOR.md` |
| 核心铁律 | `SPEC/00-PHILOSOPHY.md` |
| auto_graph→CompilerGraph 直接构建 | `SPEC/01-REQUIREMENTS.md` REQ-UGS-001 |
