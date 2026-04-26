# 计算图中间表示

> **SSOT**: 本文档是 gllm + gllm-kernels 统一项目计算图 IR（中间表示）的唯一真源。定义三种图表示、FusedOp 高层语义、优化 Pass 管线、SymDim 动态维度系统和 FusedGraphExecutor。
>
> 交叉引用: `01-JIT-PIPELINE.md`（JIT 管线、StrategySelector §5、FusionStrategy kernel 级融合、CompilerOp 原子算子、FusedOp→CompilerOp 展开映射）、`02-HARDWARE.md`（DeviceProfile）、`04-OPERATORS.md`（算子库与硬件特化）

## 1. 三种图表示

系统通过三种图表示连接 YAML 模板/ONNX 输入到 JIT 编译器的内部 IR:

```
YAML Template -> OnnxGraph -> CompilerGraph -> FusedGraph
   (架构描述)     (统一输入)      (JIT 内部 IR)     (可执行图)
```

| 表示 | 用途 | 生命周期 | 定义位置 |
|------|------|---------|---------|
| **OnnxGraph** | 统一输入格式，兼容 YAML 模板和 ONNX 解析结果 | 模型加载时构建，编译后释放 | `src/graph/types.rs` |
| **CompilerGraph** | JIT 编译器内部 SSA IR，SymExec 的消费格式 | 编译期间使用 | `gllm-kernels/src/compiler/graph.rs` |
| **FusedGraph** | 融合后的可执行图，绑定到 FusedGraphExecutor | 推理期间常驻 | `src/graph/types.rs` |

### 1.1 OnnxGraph

OnnxGraph 是 YAML 模板和 ONNX 解析结果的统一输入格式。包含节点、初始值、输出信息和命名张量边。

```rust
pub struct OnnxGraph {
    pub nodes: Vec<OnnxNode>,
    pub initializers: HashMap<String, TensorMeta>,
    pub outputs: Vec<String>,
}
```

### 1.2 CompilerGraph

CompilerGraph 是 JIT 四阶段管线的核心数据结构。张量通过 `SymDim` 表达形状，算子通过 `CompilerOp` 表达语义。`CompilerOp` 完整定义和 FusedOp→CompilerOp 展开映射见 `SPEC/01-JIT-PIPELINE.md` §8。

```rust
pub struct CompilerGraph {
    pub ops: Vec<CompilerOp>,
    pub tensors: HashMap<TensorId, TensorDesc>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

/// 张量描述 — 携带量化元数据供 StrategySelector 使用 (ARCH-COMPUTE-TIME-DEQUANT)
pub struct TensorDesc {
    pub shape: Vec<SymDim>,
    pub dtype: DType,                // 计算精度 (F32 累加)
    pub source_quant: QuantType,     // 原始量化格式 (Q4_K / BF16 / F32 原生 / MXFP4)
    pub block_size: Option<usize>,   // 量化块大小 (MXFP4=32, Q4_K=256, F32=None)
}
```

**量化元数据流 (ARCH-COMPUTE-TIME-DEQUANT)**:

```
Loader: GGUF Q4_K → QuantType::Q4_K { block_size: 256 }
    ↓
gllm: OnnxGraphConverter → TensorDesc { dtype: F32, source_quant: Q4_K, block_size: Some(256) }
    ↓
gllm-kernels: StrategySelector 读 TensorDesc.source_quant
    → 匹配 DequantComputeVariant::FusedQ4Gemm { block_size: 256 }
    ↓
Phase 3: 生成融合 "读 4bit → 解包 → F32 乘累加" 代码
```

| source_quant | JIT 行为 |
|-------------|---------|
| `F32` | 直接计算，无 dequant 步骤 |
| `BF16` | 硬件有 BF16 指令 (VDPBF16PS / BF16 WMMA) → 原生路径；否则 → 软件转换 |
| `Q4_K` / `Q4_0` | FusedQ4Gemm: 微核内循环解包 4bit + scale → 乘累加 |
| `Q8_0` / `Q8_K` | FusedQ8Gemm: 微核内循环 8bit × scale → 乘累加 |
| `MXFP4` | FusedMxfp4Gemm: 双指针 (blocks + scales) 读取 |
| `Q4_K × Q8_0` on GFX12 | Rdna4WmmaA4W8: 硬件隐式 dequant，零软件开销 |

### 1.3 FusedGraph

FusedGraph 是优化 Pass 固化后的最终执行图。包含融合节点、权重绑定、量化标注和稀疏张量绑定。

```rust
pub struct FusedGraph {
    pub nodes: Vec<FusedNode>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub weight_bindings: HashMap<String, WeightBinding>,
    pub quantization_info: HashMap<String, QuantizationInfo>,
    pub sparse_tensors: HashMap<String, SparseTensorBinding>,
    pub stats: OptimizationStats,
    pub weight_aliases: HashMap<String, Vec<String>>,
}
```

### 1.4 FusedOp 枚举

融合算子的语义枚举。每个变体携带完整的配置 struct，对应一个完整的计算子图（如注意力、FFN）。

```rust
pub enum FusedOp {
    FlashAttention(FlashAttentionConfig),
    SwiGLU(SwiGLUConfig),
    RoPE(RoPEConfig),
    FusedQkvRope(FusedQkvRopeConfig),
    FusedRMSLinear(FusedRMSLinearConfig),
    GQA(GQAConfig),
    MoERouting(MoERoutingConfig),
    FusedEmbedRerank,                          // Embedding+Rerank 融合
    Atomic(AtomicOp),                          // 未融合的底层操作
}
```

#### 配置结构

```rust
pub struct FlashAttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale: Option<f32>,       // None 时使用 1/sqrt(head_dim)
    pub causal: bool,             // 因果掩码
}

pub struct SwiGLUConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

pub struct RoPEConfig {
    pub head_dim: usize,
    pub rope_theta: f64,
    pub max_seq_len: usize,
    pub interleaved: bool,
}

pub struct FusedQkvRopeConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
}

pub struct FusedRMSLinearConfig {
    pub hidden_size: usize,
    pub eps: f32,
}

pub struct GQAConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
}

pub struct MoERoutingConfig {
    pub num_experts: usize,
    pub top_k: usize,
    pub capacity_factor: f32,
}

pub struct AtomicOp {
    pub op_type: String,          // ONNX 算子类型名
}
```

| FusedOp | 语义 | 子图组成 |
|---------|------|---------|
| `FlashAttention` | 融合注意力 | Softmax + 因果掩码 + 输出投影 + KV cache 管理 |
| `GQA` | 分组查询注意力 | KV cache 读取 + GQA 分组 + 缩放 |
| `SwiGLU` | 融合 FFN | SiLU(gate) * up + down 投影 + 输出投影 |
| `RoPE` | 旋转位置编码 | Q/K 向量旋转 |
| `FusedQkvRope` | 融合 QKV+RoPE | QKV 投影 + RoPE 合并 |
| `MoERouting` | MoE 路由 | Gate + 专家选择 + 加权聚合 |
| `FusedRMSLinear` | 融合 RmsNorm+线性 | RmsNorm + GEMM 融合 |
| `FusedEmbedRerank` | 融合 Embedding+Rerank | Embedding 模型 hidden state → Cross-Attention Rerank → 条件输出 embedding |
| `Atomic` | 原子算子 | 未融合的单个底层操作 |

**铁律**: `Atomic` 是 SPEC 唯一授权的降级。仅在无法匹配融合模式时允许（ARCH-ONNX）。参见 CLAUDE.md NO_SILENT_FALLBACK。

`Reshape` / `Transpose` 是纯元数据操作，JIT codegen 返回 `Ok(())`（NOP）——详见 `SPEC/00-PHILOSOPHY.md`。

## 2. 优化 Pass 管线

优化 Pass 按 Pipeline 模式串联执行，顺序固定: PatternFusion -> HardwareFusion -> ConstantFolding -> DCE。

所有 Pass 实现 `Pass` trait:

```rust
pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, graph: &FusedGraph) -> Result<FusedGraph, Error>;
}
```

| Pass | 职责 | 实现位置 |
|------|------|---------|
| PatternFusionPass | 子图模式匹配 → FusedOp | `src/graph/optimizer/pattern_fusion.rs` |
| HardwareFusionPass | 硬件约束检查 → 条件降级 | `src/graph/optimizer/hardware_fusion.rs` |
| ConstantFolding | 编译时常量计算 → 结果嵌入图 | `src/graph/optimizer/constant_folding.rs` |
| DeadCodeElimination | 移除未使用节点和张量 | `src/graph/optimizer/dead_code.rs` |

### 2.1 PatternFusion（模式融合）

通过子图模式匹配将算子融合为高层 FusedOp。这是图级的模式匹配，与 JIT kernel 级的 `FusionStrategy`（`01-JIT-PIPELINE.md` §4.3）是不同抽象层：

- **图级 PatternFusion**（本节）: ONNX 子图 → FusedOp 语义枚举
- **Kernel 级 FusionStrategy**（`01-JIT-PIPELINE.md`）: CompilerOp 序列 → 指令级融合策略

| 子图模式 | 匹配规则 | 融合结果 FusedOp |
|---------|---------|-----------------|
| MatMul + Add + Gelu/SiLU | GEMM 后接 bias + 激活 | `FusedRMSLinear`（带激活变体） |
| MatMul + Add | GEMM 后接 bias | `FusedRMSLinear` |
| RmsNorm + Linear/MatMul | Norm 输出直通 GEMM 输入 | `FusedRMSLinear` |
| RmsNorm + GEMM | 同上 | `FusedRMSLinear` |
| QKV 投影 + RoPE | Q/K/V MatMul 后接旋转 | `FusedQkvRope` |
| Attention 子图 | QK^T → Softmax → AV → 输出投影 | `FlashAttention` / `GQA` |
| Gate + SiLU * Up + Down | SwiGLU FFN 三段式 | `SwiGLU` |
| Gate + TopK + 专家选择 | MoE 路由子图 | `MoERouting` |

### 2.2 HardwareFusion（硬件感知融合）

根据 `DeviceProfile` 检查硬件约束。仅在约束违反时降级到 Standalone 模式。这是 SPEC 唯一授权的降级。

硬件参数通过 `DeviceProfile` 一次性探测、编译时确定、运行时固定。JIT codegen 为当前硬件生成最优路径。

架构原则:

```
错误: FusionRule 碰到硬件能力不足 -> 降级为 Atomic -> executor 拆分执行
正确: FusionRule 生成融合图 -> JIT codegen 根据 DeviceProfile 生成最优机器码
```

硬件感知融合策略（EpilogueInjection 深度、TileLevelFusion 阈值等）详见 `SPEC/04-OPERATORS.md` §8。

### 2.3 ConstantFolding（常量折叠）

编译时计算常量表达式并将结果嵌入图。消除冗余计算节点。

### 2.4 DeadCodeElimination（死代码消除）

移除未使用的节点和张量。优化后图中不存在出度为 0 的非输出节点（无 dangling edges）。

## 3. SymDim 动态维度系统

```rust
pub enum SymDim {
    Concrete(usize),
    Symbolic(String),
}
```

张量形状可以混合具体维度和符号维度:

```rust
// hidden_size 已知: SymDim::Concrete(hidden_size)
// total_seq, 运行时绑定: SymDim::Symbolic("total_seq")
graph.add_tensor("kv_cache", &[
    SymDim::Concrete(num_layers),
    SymDim::Concrete(kv_dim),
]);
```

运行时绑定 `ShapeBinding`:

```rust
pub struct ShapeBinding {
    pub bindings: HashMap<String, usize>,
}
```

**铁律**:
- 禁止在图构建时将 `total_seq`、`batch_size` 等动态维度硬编码为具体数值
- 每步 decode 递增 `total_seq` 时禁止重新编译图
- `Concrete` 维度在编译时参与代码生成优化（循环展开/tile 大小）；`Symbolic` 维度在运行时从 binding 读取，生成通用循环
- 缓存粒度 = 全层融合图，编译一次，绑定 `ShapeBinding`

> **穿透协议**: SymDim 从 FusedGraph 到 JIT 机器码的完整穿透机制见 `DOCS/scheduling/symdim-threading-protocol.md` (ARCH-SYMDIM-THREADING)。
> 核心要求：SymDim::Symbolic 信息必须穿透 plan_lower → lower → VmInstr，到 BoundExpr::Symbolic 自然转换为运行时循环 bound。禁止 `as_concrete().unwrap_or()` 丢弃符号信息。

## 4. FusedGraphExecutor

提供两种执行路径:

| 方法 | 用途 |
|------|------|
| `run_with_kv_cache(...)` | 标准推理路径，CallbackChain 为空时零额外开销 |
| `run_with_kv_cache_with_callbacks(..., callbacks)` | 优化模块接入路径（支持层跳过/注入/提前退出） |

### 4.1 节点执行模型

```
for (node_idx, node) in graph.nodes.iter().enumerate() {
    let action = callbacks.pre_node(ctx: node_idx, node, hidden_state)?;
    match action {
        Continue => {
            execute_fused_op(node, hidden_state, kv_cache, binding);
            callbacks.post_node(node_idx, node, output);
        }
        SkipThisNode => { /* GateSkip, ResidualBypass */ }
        ExitEarly { logits } => break loop
        InjectHidden { data } => { hidden_state = data }
    }
}
```

**零开销保证**: 当 CallbackChain 为空时，`run_with_kv_cache()` 走原始路径，无额外分支。编译器会将空 chain 检查优化为死代码。

CallbackAction 和 CallbackChain 完整定义见 `SPEC/05-OPTIMIZATIONS.md` §1 Per-Node Callback 架构。

## 5. 交叉引用

| 主题 | 位置 |
|------|------|
| JIT 四阶段管线 + Kernel 级 FusionStrategy | `SPEC/01-JIT-PIPELINE.md` |
| StrategySelector 硬件策略选择 | `SPEC/01-JIT-PIPELINE.md` §5 |
| ComputeStrategy 枚举族 (GemmVariant/AttentionVariant/...) | `SPEC/01-JIT-PIPELINE.md` §5.3-5.10 |
| CompilerOp 原子算子 + FusedOp→CompilerOp 映射 | `SPEC/01-JIT-PIPELINE.md` §8 |
| 策略选择与融合决策交互规则 | `SPEC/05-OPTIMIZATIONS.md` §12 |
| 算子覆盖率审计 | `SPEC/05-OPTIMIZATIONS.md` §13 |
| 硬件探测与 DeviceProfile | `SPEC/02-HARDWARE.md` |
| 算子库 + 硬件特化路径 + 融合策略差异矩阵 | `SPEC/04-OPERATORS.md` |
| Per-Node Callback 架构 | `SPEC/05-OPTIMIZATIONS.md` §1 |
| 核心铁律 | `SPEC/00-PHILOSOPHY.md` |
