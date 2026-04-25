# 四阶段 JIT 编译管线

> **SSOT**: 本文件定义 gllm-kernels JIT 编译器的四阶段管线架构、融合规则、缓存协议。硬件探测与 DeviceProfile 定义见 `SPEC/02-HARDWARE.md`。核心铁律见 `SPEC/00-PHILOSOPHY.md`。

## 1. 管线总览

算子的唯一定义来源是 `extern "C"` 纯标量函数。编译器通过二进制符号执行自动提取计算结构（OpTrace），然后根据 DeviceProfile 生成最优融合 SIMD/GPU 代码。

```
标量函数注册表 (ScalarOpRegistry)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 0: 标量参考实现                                        │
│  · extern "C" 纯标量函数 → iced-x86 反汇编 → 符号执行        │
│  · 输出: OpTrace (ComputePattern + TraceOp SSA 序列)         │
│  · 首次分析后缓存 (OpKind → OpTrace HashMap)                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
CompilerGraph + DeviceProfile + OpTrace 缓存
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 1: 语义 DAG 构筑 (SemanticDAG)                        │
│  · OpTrace.pattern → OpClass 自动推导                        │
│  · 张量 def-use 链 + 后支配树                                │
│  · 瓶颈分析 (Compute / Memory / Mixed)                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 2: Profile-Driven 融合决策 (FusionPlan)               │
│  · 9 条 FusionRule + DeviceProfile 约束检查                  │
│  · 分块配置 + 并行策略 + Buffer 规划                          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 3: ISA Lowering (代码生成)                            │
│  · x86_64: iced-x86 CodeAssembler                           │
│  · AArch64: dynasm-rs Assembler                             │
│  · GPU: PtxCodeGen / HipCodeGen / AirCodeGen                │
│  · 输出: MegaKernel binary (mmap RWX 或 GPU module)              │
└──────────────────────────────────────────────────────────────┘
```

## 2. Phase 0: 标量参考实现

### 2.1 ScalarOpRegistry

所有算子以 `extern "C"` 纯标量函数注册。`fn_ptr` 同时是执行入口（golden reference）和分析入口（符号执行反汇编起点）。

```rust
pub struct ScalarOpRegistry {
    entries: HashMap<OpKind, ScalarFnSignature>,
    trace_cache: HashMap<OpKind, OpTrace>,
}

impl ScalarOpRegistry {
    pub fn register(&mut self, op: OpKind, sig: ScalarFnSignature);
    pub fn get_trace(&mut self, op: &OpKind) -> Result<&OpTrace, CompileError>;
}
```

### 2.2 标量函数约束

编译器可分析的前提：

- `extern "C"` ABI — 无 Rust name mangling
- 只用标量算术：`+`, `-`, `*`, `/`, `exp()`, `sqrt()`, `tanh()`
- 不调用其他自定义函数（libm 除外）
- 不做堆分配
- 循环结构清晰：`for i in 0..n` 或等价 while 循环
- 编译时 `-C opt-level=1`（保留循环结构，不做向量化）

### 2.3 标量函数示例

```rust
/// SiLU: out[i] = x[i] / (1 + exp(-x[i]))
#[no_mangle]
pub extern "C" fn scalar_silu(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        unsafe {
            let v = *x.add(i);
            *out.add(i) = v / (1.0 + (-v).exp());
        }
    }
}

/// RMSNorm: two-pass — sum_squares then scale
#[no_mangle]
pub extern "C" fn scalar_rms_norm(
    x: *const f32, weight: *const f32, out: *mut f32, n: usize, eps: f32,
) {
    unsafe {
        let mut ss: f32 = 0.0;
        for i in 0..n { let v = *x.add(i); ss += v * v; }
        let inv_rms = 1.0 / ((ss / n as f32) + eps).sqrt();
        for i in 0..n { *out.add(i) = *x.add(i) * inv_rms * *weight.add(i); }
    }
}
```

### 2.4 禁止运行时调用 (NO_SCALAR)

`scalar_ops.rs` 中的函数仅作为 Phase 0 的算子语义定义，供 SymExec trace 提取使用。

- 禁止在运行时推理路径中调用任何 `scalar_*` 函数
- 禁止在单元测试中调用 `scalar_*` 函数验证算子正确性
- 测试必须通过 JIT 编译图验证正确性

## 3. Phase 1: 符号执行与语义 DAG

### 3.1 符号执行流程

1. **反汇编**: iced-x86 Decoder 从函数地址开始反汇编到 `ret`
2. **循环检测**: backward jump = back-edge = 循环
3. **符号执行**: `SymState`（寄存器 → `SymValue` 映射）逐指令推进
4. **libm 调用识别**: `call` 目标解析为 `expf`, `sqrtf`, `tanhf` 等
5. **归约模式识别**: 循环内 `addss xmm_acc, xmm_temp` 跨迭代存活 → 归约累加器
6. **GEMM 模式识别**: 三重嵌套循环 + FMA 累加 → `ComputePattern::Gemm`

### 3.2 OpTrace

Phase 0 输出的完整计算结构描述：

```rust
pub struct OpTrace {
    pub op_kind: OpKind,
    pub pattern: ComputePattern,
    pub signature: ScalarFnSignature,
}
```

### 3.3 ComputePattern

从标量函数的循环结构和数据流自动识别：

| ComputePattern | 识别特征 | OpClass 推导 |
|----------------|---------|-------------|
| `Elementwise { body }` | 单循环，load 1 → compute → store 1 | ElemWise |
| `BinaryElementwise { body }` | 单循环，load 2 → compute → store 1 | ElemWise |
| `Injective { body, num_inputs, num_outputs }` | 多输入/多输出逐元素变换 | Injective |
| `Reduction { identity, combine }` | 循环内累加器跨迭代存活 | Reduction |
| `NormLike { reduce, finalize, transform }` | 两个连续循环，第二循环使用第一循环归约结果 | Reduction |
| `Gemm` | 三层嵌套循环 + FMA 累加 | Gemm |
| `QuantDecode { block_size, decode }` | 块级循环 + 位操作解包 | Opaque |

### 3.4 TraceOp (SSA)

Phase 3 代码生成时，每个 TraceOp 映射到对应 SIMD/GPU 指令：

```rust
pub enum TraceOp {
    // ── 数据访问 ──
    Input(u32), Const(f64),
    // ── 二元算术 ──
    Add(u32, u32), Sub(u32, u32), Mul(u32, u32), Div(u32, u32),
    Fma(u32, u32, u32),
    Max(u32, u32), Min(u32, u32),
    // ── 一元算术 ──
    Neg(u32), Abs(u32),
    Exp(u32), Sqrt(u32), Rsqrt(u32), Tanh(u32), Recip(u32), Log(u32),
    // ── 比较与类型转换 (ARCH-AUTO-INSTR-SELECT 待实现) ──
    Compare { a: u32, b: u32, op: CmpOp },  // P0: 解锁条件分支
    Cast { src: u32, target_dtype: DType },  // P0: 解锁 dtype 转换
    // ── 横向归约 (ARCH-AUTO-INSTR-SELECT 待实现) ──
    HReduce { src: u32, op: ReduceOp, width: usize },  // P1: 解锁 softmax/norm 全自动
    // ── 控制流 (ARCH-AUTO-INSTR-SELECT 待实现) ──
    ConditionalBranch { cond: u32, true_body: Vec<TraceOp>, false_body: Vec<TraceOp> },  // P1
    // ── 向量重排 (P2) ──
    Permute { src: u32, indices: Vec<u32> },
    MaskedLoad { base: u32, mask: u32, offset: Offset },
    MaskedStore { base: u32, src: u32, mask: u32, offset: Offset },
}
```

**TraceOp 扩展解锁矩阵**:

| TraceOp | 解锁的算子 | 对应的 VmInstr |
|---------|-----------|---------------|
| Compare | if-else 条件, SelectOp | VecCmp |
| Cast | dtype 转换 (f32↔f16↔bf16) | VecCast |
| HReduce | softmax, norm 归约 (sum/max) | VecReduce |
| ConditionalBranch | 条件执行路径 | 条件 JMP |
| Permute | 向量重排 (shuffle) | VecPermute |
| MaskedLoad/Store | 尾部掩码, 不对齐访问 | MaskedLoad/Store |

### 3.5 SemanticDAG

CompilerGraph + 语义标注 + 数据流分析：

```rust
pub struct SemanticDAG {
    pub nodes: Vec<SemanticNode>,
    pub tensor_edges: Vec<TensorEdge>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub post_dominator_tree: PostDomTree,
}
```

每个 `SemanticNode` 携带 `OpTrace`（完整计算结构）、`OpClass`、`Bottleneck`（Compute/Memory/Mixed）、算术强度等。后支配树用于 Phase 2 融合组划分。

## 4. Phase 2: 融合决策与硬件约束

### 4.1 入口签名

```rust
pub fn fuse(graph: &CompilerGraph, profile: &DeviceProfile) -> FusionPlan;
```

融合决策由 `HwOptEngine`（`SPEC/02-HARDWARE.md` §10）统一求解的 `HwOptPlan.fusion` 驱动。`HwOptEngine` 内部消费 `DeviceProfile` + `CompilerConstraints` + `ProbeResult`，通过 CostModel 评估候选策略，输出不可变 `HwOptPlan`。本阶段不再直接读取 `DeviceProfile` 做策略决策。

### 4.2 融合组划分规则

基于 TVM 融合规则，在后支配树上从叶子向根遍历：

| 生产者 | 消费者 | 融合规则 |
|--------|--------|---------|
| ElemWise / Injective | 任意 | 可融合进消费者 |
| Reduction | ElemWise / Injective | 消费者作为 epilogue |
| Gemm | ElemWise | Epilogue Injection |
| Gemm | Reduction | 不融合 |
| Opaque | 任意 | 不融合 |

额外约束：
- 生产者有多个消费者 → 不融合
- 融合后寄存器压力 > `profile.num_simd_regs()` → 拒绝
- Epilogue scratch 寄存器 > 微内核剩余 → 拒绝 epilogue injection

### 4.3 九种融合模式

```rust
pub enum FusionStrategy {
    Single,
    LoopFusion { chain_nodes: Vec<usize> },
    EpilogueInjection { gemm_node: usize, epilogue_ops: Vec<EpilogueOp> },
    TileLevelFusion { gemm_node: usize, tiled_predecessor: usize, tile_rows: usize, epilogue_ops: Vec<EpilogueOp> },
    ComputeRoot { predecessor: usize, gemm_node: usize },
    QkvSharedInput,
    NormIntoGemm,
    FFNBlock,
    CrossLayerResidual,
}
```

### 4.4 EpilogueInjection

GEMM 微内核累加完毕后，取消费者算子的 `OpTrace.body`（Vec<TraceOp>），对每个 TraceOp 生成 SIMD 指令，在累加器寄存器上原地执行。数据不落地内存。

```
未融合: GEMM 写回 → Bias 读+写 → SiLU 读+写 = 3 次写 + 2 次读
融合后: GEMM K-loop → [累加器原地执行 Bias+SiLU] → 1 次写
```

### 4.5 LoopFusion

多个 elemwise 算子合并为单循环，数据在寄存器中流过整个链：

```
未融合: SiLU 写 → VecMul 读+写 → VecAdd 读+写 = 3 读 + 3 写
融合后: 单循环 ymm0 = silu(load) * load → +load → store = 3 读 + 1 写
```

### 4.6 TileLevelFusion vs ComputeRoot

由 DeviceProfile 的 cache 层级数据驱动：

```
if predecessor_output_bytes > profile.l1_cache_bytes * 0.75 {
    // 输出超过 L1 75% → 先算完再读已被逐出
    // 必须嵌入 GEMM MC 循环，每次只算 MC 行
    TileLevelFusion { tile_rows: mc }
} else {
    // 输出 ≤ L1 75% → 先算完，结果整体留在 L1
    ComputeRoot
}
```

TileLevelFusion scratch buffer: MC 行的前驱结果写入 scratchpad 的 normed 区域，紧接着被 pack_a 消费。前驱算子逐行独立，按 MC 行切分不影响正确性。

### 4.7 QkvSharedInput

Q/K/V 三个 GEMM 共享 pack_a（输入矩阵相同），消除 3 次重复 pack。

### 4.8 NormIntoGemm

RmsNorm 输出直接喂入 GEMM（无中间写回）。与 TileLevelFusion 的区别：NormIntoGemm 是 Norm 输出直接作为 GEMM 输入的语义级融合。

### 4.9 FFNBlock

Gate GEMM + Up GEMM 的 Gate/Up 结果融合激活（如 SiLU）+ 乘法，合并为单次计算块。

### 4.10 CrossLayerResidual

Add → RmsNorm scratchpad 直通融合。残差连接的 Add 结果直接传入下一层 RmsNorm，避免中间写回。

### 4.11 融合输出

```rust
pub struct FusionPlan {
    pub groups: Vec<FusionGroup>,
    pub tile_configs: Vec<TileConfig>,
    pub buffer_plan: BufferPlan,
}
```

`TileConfig` 包含 GEMM BLIS 三级分块（KC/MC/NC 适配 L1/L2/L3）+ 并行策略 + 预取距离。`BufferPlan` 通过张量活性分析 + 区间图着色贪心算法最大化 buffer 原地复用。

## 5. Phase 3: ISA Lowering

### 5.1 自动指令选择器 (Auto Instruction Selection)

#### 5.1.1 设计原则

Phase 3 的 TraceOp → VmInstr → ISA 映射必须是**完全自动的**，类似 LLVM SelectionDAG。禁止在 `emit_standalone_op` 中手写 `OpKind::*` match arm。

```
正确架构:
  Scalar → SymExec → TraceOp → [自动指令选择] → VmInstr → ISA Lowering → Machine Code
                                      ↑
                                 类似 LLVM SelectionDAG
                                 由 ComputePattern 驱动，算法保证正确性
```

**错误架构（已废弃）**:
```
  Scalar → SymExec → TraceOp → [plan_lower.rs 手写 emit] → VmInstr → ISA Lowering
                                      ↑
                                 人肉指令选择，bug 源头
```

#### 5.1.2 自动分发规则

`emit_standalone_op` 必须按 `ComputePattern`（或等价的 `OpSemantics`）分类分发，不允许逐个 `OpKind` 手写 match arm：

| ComputePattern | OpSemantics | 分发策略 | 自动化级别 |
|----------------|-------------|---------|-----------|
| Elementwise { body } | Elementwise | `auto_lower_trace(body)` → VmInstr | **全自动** — 零手写代码 |
| BinaryElementwise { body } | Elementwise | `auto_lower_trace(body)` → VmInstr | **全自动** — 零手写代码 |
| NormLike { reduce, finalize, transform } | Reduction | 专用 `lower_norm*` 函数 | 半自动 — trace 驱动 norm 公式 |
| Gemm | Gemm | 专用 `lower_gemm*` 函数 | 半自动 — 硬件分块策略 |
| Opaque (结构/控制) | Opaque | 专用 lower 函数或 NOP | 手写 — 控制流指令 |

#### 5.1.3 自动分发实现要求

1. **函数入口**: `emit_standalone_op` 函数顶部首先尝试 `try_auto_dispatch_elementwise()`
2. **ComputePattern 路由**: elementwise 失败后，按 ComputePattern 类型路由到专用 lower 函数
3. **禁止手写 OpKind match**: 不允许 `OpKind::Xxx => { ... }` 形式的逐算子分发
4. **新增算子**: 注册 scalar impl → SymExec 提取 trace → elementwise 自动完成；复杂算子写一个 `lower_*` 函数 + 注册 ComputePattern 路由
5. **缺失算子**: 走到 catch-all 时返回 `Err`，禁止静默 NOP（NO_SILENT_FALLBACK）

#### 5.1.4 TraceOp → VmInstr 自动查表

`auto_lower_trace()` 实现完全自动的 TraceOp → VmInstr 映射，每个 TraceOp 变体自带映射语义：

| TraceOp | VmInstr | 辅助函数 |
|---------|---------|---------|
| Input(n) | 返回 inputs[n] | — |
| Const(val) | Broadcast { ScalarExpr::Const(val) } | — |
| Add(a,b) | VecBinOp { op: Add } | `emit_binop()` |
| Sub/Mul/Div/Max/Min | VecBinOp { op: ... } | `emit_binop()` |
| Neg/Abs/Sqrt/Rsqrt/Recip | VecUnaryOp { op: ... } | `emit_unary()` |
| Fma(a,b,c) | Fma { ... } | — |
| Exp/Tanh/Log | Transcendental { func: ... } | `emit_transcendental()` |
| HReduce(a, op) | VecReduce { op } | `emit_hreduce()` (待实现) |
| Compare(a, b, op) | VecCmp { op } | `emit_cmp()` (待实现) |
| Cast(a, dtype) | VecCast { dtype } | `emit_cast()` (待实现) |

**关键保证**: 新增 TraceOp 变体只需在 `auto_lower_trace` 的 match 中添加**一行**映射，不需要修改任何其他 lower 函数。

#### 5.1.5 待扩展的 TraceOp（优先级排序）

当前缺少的 TraceOp 变体阻止了 softmax/norm 的全自动 lowering：

| 优先级 | TraceOp | 需要的 VmInstr | 解锁的算子 |
|--------|---------|---------------|-----------|
| P0 | Compare | VecCmp | 条件分支、if-else |
| P0 | Cast | VecCast | dtype 转换 |
| P1 | HReduce | VecReduce + VecStore | softmax、norm 归约 |
| P1 | ConditionalBranch | 条件 JMP | 控制流 |
| P2 | Permute | VecPermute | 向量重排 |
| P2 | MaskedOp | MaskedLoad/Store | 掩码操作 |

### 5.1.6 TraceOp → ISA 指令映射

Phase 3 从 OpTrace 的 `Vec<TraceOp>` 直接映射到平台指令。不硬编码算子语义。

**x86_64 (iced-x86)**:

| TraceOp | AVX2 (ymm) / AVX-512 (zmm) |
|---------|----------------------------|
| Add(a,b) | vaddps |
| Mul(a,b) | vmulps |
| Fma(a,b,c) | vfmadd231ps |
| Exp(a) | 多项式逼近 ~12 条 |
| Sqrt(a) | vsqrtps |
| Recip(a) | vrcpps + Newton |

**AArch64 (dynasm-rs)**:

| TraceOp | NEON (v.4s) |
|---------|-------------|
| Add(a,b) | fadd |
| Fma(a,b,c) | fmla |
| Exp(a) | 多项式逼近 |
| Sqrt(a) | fsqrt |

**GPU (PTX)**:

| TraceOp | PTX |
|---------|-----|
| Add | add.f32 |
| Fma | fma.rn.f32 |
| Exp | ex2.approx.f32 + 多项式修正 |

### 5.2 PlatformBackend 统一接口

Phase 1-2 完全平台无关。平台差异封装在 `MachineCodeEmitter` trait 中。

```rust
pub trait PlatformBackend {
    type Emitter: MachineCodeEmitter;
    fn new_emitter(&self) -> Self::Emitter;
    fn platform(&self) -> Platform;
    fn num_simd_regs(&self) -> usize;
}
```

微内核规格（MR×NR）、GEMM 代码结构、融合策略硬件差异矩阵、算子实现硬件差异矩阵详见 `SPEC/04-OPERATORS.md` §8-10。

### 5.3 禁止事项

- 禁止跳过 Lifting 阶段直接手写 ISA 汇编（如直接写 AVX2 Softmax）
- 禁止在 JIT 代码中通过 `call` 指令调用预编译的 Rust/C 函数
- 禁止写死特定 ISA，必须通过 DeviceProfile 参数化
- 禁止 emit_nop() 作为未实现 op 的 catch-all（见 `00-PHILOSOPHY.md` NO_SILENT_FALLBACK）
- 硬件差异体现在 codegen 层的指令选择，不是 fusion 层的算子拆分（见 `00-PHILOSOPHY.md` NO_HW_DEGRADATION）

## 6. JIT 缓存协议

### 6.1 编译时机

- 编译只发生在模型加载时
- 推理热路径（decode step 层循环）中禁止任何编译行为
- `InferenceCompiler::new()` / `compile_graph()` / `build_*_graph()` 只在模型加载时调用

### 6.2 SymDim 动态维度

动态维度（seq_len, total_seq）通过 `SymDim::Symbolic` + `ShapeBinding` 运行时绑定，不触发重编译。

```rust
pub enum SymDim {
    Concrete(usize),
    Symbolic(String),
}

pub struct ShapeBinding {
    pub bindings: HashMap<String, usize>,
}
```

铁律：
- 禁止在图构建时将 `total_seq`、`batch_size` 等动态维度硬编码为具体数值
- 禁止因 shape 变化而重新调用 `compile_and_run` 构建新图
- 编译一次，通过 `ShapeBinding` 在每个 decode step 传入当前值

### 6.3 三级缓存

| 级别 | 作用域 | 存储位置 | 生命周期 |
|------|--------|---------|---------|
| L1 | 模型级 | 内存 | 模型卸载时释放 |
| L2 | 全局 LRU | 内存 | 进程退出时释放 |
| L3 | 持久化 | `~/.gllm/jit_cache/` | 7 天 TTL 自动清理 |

Debug 模式（`cargo test` / `cargo build`）: L3 磁盘缓存完全禁用。
Release 模式（`cargo build --release`）: L3 正常工作。

### 6.4 CPU/GPU 统一 (ARCH-CPU-GPU-UNIFIED)

CPU 和 GPU 后端共享完全一致的 `CompilerGraph` IR 和 `GraphType`。

- 禁止以 `Cpu`/`Gpu`/后端名称为前缀的 `GraphType` 变体（如 `CpuDecoderLayer`）
- 禁止 CPU 专用图构建器（如 `build_decoder_layer_graph()`）
- 禁止子算子级 `GraphType`（如 `KvProjection`、`QRope`）
- CPU 和 GPU 给定相同输入和权重，输出必须在浮点精度范围内一致

### 6.5 编译粒度

缓存粒度 = 全层融合图：

- `FusedAttentionLayer`（RmsNorm + Q/K/V GEMM + RoPE + Attention + O GEMM）
- `FusedFfnLayer`（RmsNorm + Gate/Up GEMM + Activation + Down GEMM）

禁止子算子级缓存（如单独缓存一个 RmsNorm 或一个 GEMM）。

## 7. CompilerGraph 接口边界

gllm 负责将高层 FusedGraph 展开为原子算子 DAG 后传入 gllm-kernels。

```
gllm: ONNX Graph → GraphOptimizer → FusedGraph → expand_for_compiler() → CompilerGraph
gllm-kernels: compile_graph(graph, profile) → MegaKernel binary
```

### CompilerGraph

```rust
pub struct CompilerGraph {
    pub nodes: Vec<CompilerNode>,
    pub inputs: Vec<TensorDesc>,
    pub outputs: Vec<TensorDesc>,
}

pub enum CompilerOp {
    MatMul { m, n, k, transpose_b },
    RmsNorm { hidden_size, eps },
    LayerNorm { hidden_size, eps },
    Activation(OpKind),
    Add, Mul,
    Rope { head_dim, max_seq_len, theta },
    Softmax { axis },
    QuantMatMul { quant_type, m, n, k },
    Reshape { target_shape },
    Transpose { perm },
}
```

gllm 侧 FusedOp 到 CompilerOp 的展开规则（FusedOp 高层语义定义见 `SPEC/03-GRAPH-IR.md` §1.4）：

| FusedOp | 展开为 CompilerOp 序列 |
|---------|----------------------|
| `FlashAttention` | Reshape(Q) → Reshape(K) → Reshape(V) → MatMul(Q,K^T) → Softmax → MatMul(attn,V) → Reshape(out) |
| `GQA` | Reshape(Q) → Reshape(K) → Reshape(V) → Mul(scale) → MatMul(Q,K^T) → Softmax → MatMul(attn,V) → Reshape(out) |
| `SwiGLU` | MatMul(gate) → Activation(SiLU) → MatMul(up) → Mul → MatMul(down) |
| `RoPE` | Rope |
| `FusedQkvRope` | MatMul(Wq) → Rope → MatMul(Wk) → Rope → MatMul(Wv) |
| `MoERouting` | MatMul(gate_weight) → Softmax → TopK → [专家分支: Gather + MatMul + Mul] → Add(聚合) |
| `FusedRMSLinear` | RmsNorm → MatMul |
| `FusedEmbedRerank` | [embedder layers → L2Norm(hidden)] → embed] → [cross_attn(query_emb, doc_hidden) → score → ConditionalSelect(score, threshold)] → output embedding |
| `Atomic("Gather")` | **Gather { table_rows, embed_dim, index_dim }** — 索引查找（embedding lookup），JIT 编译为循环内索引加载 |
| `Atomic("Slice")` | **SliceView { axis, start, end }** — 零拷贝视图（指针偏移），JIT NOP（已授权 SPEC 例外） |
| `Atomic("Shape")` | 编译时常量折叠消除。output = input 恒等传递。不产生 CompilerOp |
| `Atomic(其他)` | 直接透传为单个 CompilerOp |
