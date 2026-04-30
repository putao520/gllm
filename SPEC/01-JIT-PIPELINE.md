# 四阶段 JIT 编译管线

> **SSOT**: 本文件定义 gllm-kernels JIT 编译器的四阶段管线架构、融合规则、缓存协议。硬件探测与 DeviceProfile 定义见 `SPEC/02-HARDWARE.md`。核心铁律见 `SPEC/00-PHILOSOPHY.md`。

## 0. CompilerGraph 来源 (ARCH-UNIFIED-GRAPH-SOURCE)

**铁律：CompilerGraph 的唯一来源是 gllm 侧 YAML 架构模板。gllm-kernels 不包含独立的图构建逻辑。**

```
gllm: config.json → ModelConfig → ResolvedConfig
    ↓
gllm: YAML 模板 (arch/templates/{arch}.yaml) → ArchTemplate::to_onnx_graph() → OnnxGraph
    ↓
gllm-kernels: OnnxGraphConverter::convert(onnx_graph, geometry) → CompilerGraph
    ↓
gllm-kernels: Phase 0-3 JIT 管线 (§1)
```

详细设计见 `SPEC/GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md §2.4-2.6`。

**禁止**：
- ❌ 在 `graph_builders.rs` 中手写 `decoder_model()` / `decoder_model_hetero()` / `build_layer_body()`
- ❌ 手写 `compute_per_layer_bytes()` 硬编码权重布局
- ❌ 新增模型时修改 gllm-kernels 代码（只需写 YAML 模板 + 标量算子注册）

**量化架构 (ARCH-COMPUTE-TIME-DEQUANT)**:
- 权重在加载时保持原始压缩格式 (Q4_K, Q8_0, BF16, MXFP4...)，不进行全量 dequant
- JIT 生成融合 dequant+compute 代码 (DequantComputeVariant)
- 硬件支持隐式 dequant 时 (如 GFX12 a4w8 WMMA) 优先使用硬件路径
- TensorDesc 携带 `source_quant: QuantType` 元数据，StrategySelector 据此选择策略
- 已废弃: ARCH-COMPUTE-F32 (load-time 全量 dequant 到 F32)

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
CompilerGraph + DeviceProfile + OpTrace 缓存 + TensorDesc.source_quant
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
│  Phase 2.5: StrategySelector (硬件策略选择)                  │
│  · 三路匹配: OpClass × DeviceProfile × QuantFormat           │
│  · 输出: ImplementationPlan (ComputeStrategy per group)      │
│  · 可缓存 / 可审计 / 可独立测试                               │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 3: ISA Lowering (代码生成)                            │
│  · 根据 ImplementationPlan 中的 ComputeStrategy 发射指令      │
│  · x86_64: iced-x86 CodeAssembler                           │
│  · AArch64: dynasm-rs Assembler                             │
│  · GPU: PtxCodeGen / HipCodeGen / AirCodeGen                │
│  · 输出: MegaKernel binary (mmap RWX 或 GPU module)          │
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

Phase 3 代码生成时，每个 TraceOp 映射到对应 SIMD/GPU 指令。
实现位于 `gllm-kernels/src/compiler/trace.rs`，自动指令选择器位于 `codegen/vm/auto_select.rs`。

```rust
pub enum TraceOp {
    // ── 数据访问 ──
    Input(u32),               // 加载第 i 个输入
    Const(f64),               // 浮点常量

    // ── 二元算术 (→ VecBinOp) ──
    Add(u32, u32), Sub(u32, u32), Mul(u32, u32), Div(u32, u32),
    Fma(u32, u32, u32),       // a * b + c → VmInstr::Fma
    Max(u32, u32), Min(u32, u32),

    // ── 一元算术 (→ VecUnaryOp / Transcendental) ──
    Neg(u32), Abs(u32), Sqrt(u32), Rsqrt(u32), Recip(u32),
    Exp(u32), Tanh(u32), Log(u32),
    ConditionalBranch(u32, u32, u32),  // (mask, true_val, false_val) → 条件选择

    // ── 量化混合精度 (§11 TurboQuant / §13.12 硬件拓扑) ──
    // 后端: gfx950 mfma_scale / SM100 tcgen05 / AMX-FP8 TDPFP8PS
    QuantFma {
        acc: u32, act: u32, weight: u32,
        act_dtype: QuantPrecision, weight_dtype: QuantPrecision,
    },
    BlockScale { data: u32, scale: u32, block_size: usize },  // CDNA4 mfma_scale 原生
    Cast { src: u32, from: QuantPrecision, to: QuantPrecision },  // F16C vcvtph2ps / ARM fcvtl / PTX cvt

    // ── 水平归约 (§13 Epilogue 白嫖) ──
    // 后端: x86 shuffle+hadd / ARM faddp+addv / GPU shfl.sync warp reduce
    HReduce { src: u32, op: ReduceKind },  // ReduceKind: Sum/Max/Min/Prod/Count/ArgMax

    // ── 内存层级控制 (§13.2 质心预取 / §11 TurboQuant) ──
    Prefetch { level: CacheLevel },         // prefetcht0/t1/nta / prfm / prefetch.global.L2
    NonTemporalStore,                        // vmovntps / stnp / st.global.cs

    // ── 位操作 (量化解包) ──
    BitExtract { src: u32, offset: u32, width: u32 },  // shift+mask / ubfx / bfe

    // ── 向量排列/洗牌 ──
    Permute { src: u32, indices: u32 },     // vpshufb/vpermd / tbl/tbx / prmt

    // ── 比较和掩码 (§13.1 Gate-First / §13.3 残差旁路) ──
    Compare { a: u32, b: u32, op: CmpOp },  // AVX-512 k-mask / SVE predicate / GPU predicate
    MaskedOp { op: Box<TraceOp>, mask: u32 },  // 仅对 mask=true 的 lane 执行

    // ── 原子操作 (§13.6 MoE 命中计数) ──
    AtomicAdd { addr: u32, val: u32 },       // lock xadd / ldadd / atomicAdd

    // ── 信号处理 (§11.1 TurboQuant FWHT) ──
    FWHT { src: u32, dim: usize },           // 展开的 butterfly 加减指令序列
}
```

**辅助枚举定义**:

```rust
pub enum QuantPrecision {  // 覆盖三代硬件所有精度格式
    F32, F16, BF16, TF32,       // 标准: Intel AMX-TF32
    FP8E4M3, FP8E5M2,           // NVIDIA/AMD FP8
    FP6E2M3, FP6E3M2,           // AMD CDNA4
    FP4E2M1,                    // AMD CDNA4 / NVIDIA Blackwell
    INT8, INT4, INT2, INT1,     // 整数量化 (QJL 1-bit §11.5)
}
pub enum ReduceKind { Sum, Max, Min, Prod, Count, ArgMax }
pub enum CacheLevel { L1, L2, L3, NonTemporal }
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }
```

**TraceOp → VmInstr 自动查表实现状态** (auto_select.rs):

| TraceOp | VmInstr | 辅助函数 | 状态 |
|---------|---------|---------|------|
| Input(n) | 直接返回 inputs[n] | — | ✅ |
| Const(val) | Broadcast | — | ✅ |
| Add/Sub/Mul/Div/Max/Min | VecBinOp | emit_binop() | ✅ |
| Fma | Fma | — | ✅ |
| Neg/Abs/Sqrt/Rsqrt/Recip | VecUnaryOp | emit_unary() | ✅ |
| Exp/Tanh/Log | Transcendental | emit_transcendental() | ✅ |
| Compare { .. } | VecCmp | emit_cmp() | ✅ |
| Cast { .. } | VecCast | quant_precision_bits() | ✅ |
| HReduce { .. } | HReduce | — (Sum/Max/Min/Prod) | ✅ |
| ConditionalBranch | — | — | ❌ 需 VecConditional |
| QuantFma / BlockScale | 专用 VmInstr | — | ❌ 走 StrategySelector 专用路径 |
| Prefetch / NonTemporalStore | — | — | ❌ 内存提示指令 |
| BitExtract | — | — | ❌ 量化解包专用 |
| Permute | — | — | ❌ 向量重排 |
| MaskedOp | — | — | ❌ 掩码操作 |
| AtomicAdd | AtomicAddU32 | — | ❌ MoE 专用 |
| FWHT | — | — | ❌ TurboQuant 专用 |

**TraceOp 扩展解锁矩阵**:

| TraceOp | 解锁的算子 | 对应的 VmInstr | auto_lower_trace |
|---------|-----------|---------------|------------------|
| Compare | if-else 条件, SelectOp, Gate-First 掩码 | VecCmp | ✅ 已实现 |
| Cast | dtype 转换 (f32↔f16↔bf16↔FP8) | VecCast | ✅ 已实现 |
| HReduce | softmax, norm 归约, 质心距离 | HReduce | ✅ 已实现 (Sum/Max/Min/Prod) |
| ConditionalBranch | 条件执行路径, SelectOp 替代 | VecConditional | ❌ 待实现 |
| QuantFma | 混合精度 GEMM (TurboQuant) | 专用 path via StrategySelector | ❌ 走专用 lower |
| BlockScale | 块缩放 (CDNA4 mfma_scale) | 专用 path | ❌ 走专用 lower |
| Prefetch | 质心预取, 权重预热 | PrefetchInstr | ❌ 待实现 |
| NonTemporalStore | KV 连续写入绕缓存 | NonTemporalStoreInstr | ❌ 待实现 |
| BitExtract | 量化解包 (nibbles→f32) | BitExtractInstr | ❌ 待实现 |
| Permute | 向量重排 (shuffle, RoPE) | VecPermute | ❌ 待实现 |
| MaskedOp | 尾部掩码, 不对齐访问 | MaskedLoad/Store | ❌ 待实现 |
| AtomicAdd | MoE expert 命中计数 | AtomicAddU32 | ❌ 待实现 |
| FWHT | TurboQuant 在线旋转 | butterfly 序列 | ❌ 待实现 |

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

## 5. StrategySelector (硬件策略选择)

### 5.1 定位

StrategySelector 是 Phase 2 (FusionPlan) 和 Phase 3 (ISA Lowering) 之间的显式策略选择层。它接收 FusionPlan + DeviceProfile + 量化元数据，输出 ImplementationPlan——每个融合组的最优算法变体。

```
Phase 2 FusionPlan
    +
DeviceProfile (硬件能力)
    +
TensorDesc.source_quant (每个输入张量的原始量化格式)
    ↓
StrategySelector (三路匹配: OpClass × HW × QuantFormat)
    ↓
ImplementationPlan (每个融合组的 ComputeStrategy)
    ↓
Phase 3 ISA Codegen (dumb translator — 按策略发射指令)
```

**核心原则**:
- 标量层定义数学语义 (ONE correct implementation) — 不表达硬件差异
- StrategySelector 选择算法变体 (N hardware-specific variants)
- ISA Codegen 是 "哑翻译器" — 按策略对象发射机器码，不内嵌策略决策逻辑
- 策略是一等公民对象: 可日志、可缓存、可独立测试、可 A/B 比较

### 5.2 入口签名

```rust
pub fn select_strategy(
    plan: &FusionPlan,
    profile: &DeviceProfile,
) -> ImplementationPlan;
```

量化元数据通过 `FusionPlan` → `FusionGroup` → 节点的 `TensorDesc.source_quant` 隐式提供，不需要单独参数。`DeviceProfile` 携带硬件能力矩阵（ISA 级别、tensor core 代数、WMMA 格式支持等）。

### 5.3 ImplementationPlan

```rust
pub struct ImplementationPlan {
    pub groups: Vec<GroupStrategy>,
}

pub struct GroupStrategy {
    pub fusion_group_id: usize,
    pub compute: ComputeStrategy,
    pub rationale: String,  // "GFX12 WMMA a4w8: Q4_K×Q8_0 inputs, hardware dequant"
}

pub enum ComputeStrategy {
    Gemm(GemmVariant),
    Attention(AttentionVariant),
    Norm(NormVariant),
    MoE(MoeVariant),
    Elementwise(ElementwiseVariant),
    Gather(GatherVariant),
    DequantCompute(DequantComputeVariant),
}
```

### 5.4 GemmVariant (V1-V14)

同一个数学操作 (C += A×B) 在不同硬件和量化格式下的算法变体：

| 变体 | 硬件 | 核心技术 | 输入精度 | 累加精度 |
|------|------|---------|---------|---------|
| `ScalarLoop` | 任意 (fallback) | 三重循环 | F32 | F32 |
| `Avx2Fma { mr, nr }` | x86 AVX2 | VFMA 微核, 6×4 寄存器分块 | F32 | F32 |
| `Avx512F32 { mr, nr }` | x86 AVX-512 | zmm 微核, 32 寄存器 | F32 | F32 |
| `Avx512Bf16 { mr, nr }` | x86 AVX-512 + VDPBF16PS | 原生 BF16 输入 | BF16 | F32 |
| `AmxTile { tile_m, tile_n }` | x86 AMX | TDPBF16PS 矩阵扩展 | BF16 | F32 |
| `NeonFmla { mr, nr }` | AArch64 NEON | FMLA 微核 | F32 | F32 |
| `SveFmla { mr, nr }` | AArch64 SVE | 可变长度 FMLA | F32 | F32 |
| `Sm70Wmma { stages }` | NVIDIA SM70 | WMMA tensor core v1 | F16 | F32 |
| `Sm80MmaAsync { stages, split_k }` | NVIDIA SM80 | MMA.sync + cp.async 多 stage | BF16/F16 | F32 |
| `Sm90WgmmaTma { num_waves, cluster_dims }` | NVIDIA SM90 | TMA + wgmma 协作式多 wave | BF16/F16 | F32 |
| `Sm100Tcgen05 { ... }` | NVIDIA SM100+ | tcgen05 下一代 tensor core | BF16/FP8 | F32 |
| `Rdna3WmmaF16 { ... }` | AMD RDNA 3 (gfx11) | WMMA F16 | F16 | F32 |
| `Rdna3WmmaBf16 { ... }` | AMD RDNA 3 (gfx11) | WMMA BF16 | BF16 | F32 |
| `Rdna4WmmaF16 { ... }` | AMD RDNA 4 (GFX12) | WMMA F16 | F16 | F32 |
| `Rdna4WmmaBf16 { ... }` | AMD RDNA 4 (GFX12) | WMMA BF16 | BF16 | F32 |
| `Rdna4WmmaA4W8 { ... }` | AMD RDNA 4 (GFX12) | WMMA a4w8 硬件隐式 dequant | A4+W8 | F32 |
| `Cdna2MfmaBf16 { ... }` | AMD CDNA 2 (gfx90a, MI250) | MFMA 32×32×8 BF16, Wave64 | BF16 | F32 |
| `Cdna2MfmaF16 { ... }` | AMD CDNA 2 (gfx90a, MI250) | MFMA 32×32×16 F16, Wave64 | F16 | F32 |
| `Cdna3MfmaBf16 { ... }` | AMD CDNA 3 (gfx942, MI300X) | MFMA 32×32×16 BF16, Wave64 | BF16 | F32 |
| `Cdna3MfmaF16 { ... }` | AMD CDNA 3 (gfx942, MI300X) | MFMA 32×32×32 F16, Wave64 | F16 | F32 |
| `Cdna3MfmaInt8 { ... }` | AMD CDNA 3 (gfx942, MI300X) | IMMA 32×32×32 INT8, Wave64 | INT8 | INT32 |
| `Cdna3MfmaFp8 { ... }` | AMD CDNA 3 (gfx942, MI300X) | FP8 BF8×BF8 MFMA, Wave64 | FP8 BF8 | F32 |
| `MetalSimdMM { tile_m, tile_n }` | Apple GPU 8+ (M3/M4) | `simd_matrix_multiply` 8×8×8 | FP16 | F32 |
| `MetalSimdMmBf16 { tile_m, tile_n }` | Apple GPU 9+ (M4) | `simd_matrix_multiply` BF16 | BF16 | F32 |
| `MetalTiled { tile_m, tile_n }` | Apple GPU 7 (M1/M2) | Tiled GEMM + SIMD 向量化 | FP32 | FP32 |

**多 wave 并发**: `Sm90WgmmaTma { num_waves: 2 }` 表示 2 个 wave 协作处理同一 GEMM tile。`cluster_dims` 控制 Thread Block Cluster 维度。

### 5.5 DequantComputeVariant (Fused Dequant + Compute)

**架构决策 ARCH-COMPUTE-TIME-DEQUANT**: 权重在加载时保持原始压缩格式 (Q4_K, Q8_0, BF16, MXFP4...)，JIT 生成融合 dequant+compute 代码。替代已废弃的 ARCH-COMPUTE-F32 (load-time 全量 dequant)。

| 变体 | 量化格式 | 硬件 | 机制 |
|------|---------|------|------|
| `FusedQ4Gemm { block_size }` | Q4_K / Q4_0 | CPU | 微核内循环: 读 4bit → 解包 → F32 乘累加 |
| `FusedQ8Gemm { block_size }` | Q8_0 / Q8_K | CPU | 微核内循环: 读 8bit × scale → F32 乘累加 |
| `FusedMxfp4Gemm { block_size }` | MXFP4 (blocks+scales 分离) | CPU | 双指针读取: blocks + scales → 解码 → 乘累加 |
| `FusedBf16Gemm` | BF16 | CPU AVX-512 | VDPBF16PS 原生路径 (等价 Avx512Bf16) |
| `Rdna4A4W8Wmma` | Q4×Q8 混合 | AMD GFX12 | 硬件隐式 dequant (a4w8 WMMA) — 零软件 dequant |
| `Sm80Int8TensorCore` | INT8 / UINT8 | NVIDIA SM80+ | IMMA tensor core |
| `Sm90Fp8TensorCore` | FP8 E4M3 / E5M2 | NVIDIA SM90+ | FP8 tensor core |

**内存节省对比**:

```
已废弃 (ARCH-COMPUTE-F32):
  Load: GGUF Q4_K → 全量 dequant 到 F32 → 存入 weight blob
  Compute: GEMM(F32 × F32)
  内存: elements × 4 bytes (全 F32)
  例: 4096×4096 权重 = 64 MB

当前 (ARCH-COMPUTE-TIME-DEQUANT):
  Load: GGUF Q4_K → 原始格式直接存入 weight blob (+ scale/zp 元数据)
  Compute: FusedDequantCompute(读 Q4 → 解包 → F32 乘累加)
  内存: elements × 0.5 bytes (保持 Q4) + scales
  例: 4096×4096 权重 = 8 MB + 0.5 MB scales = 8.5 MB (7.5× 节省)
```

**MXFP4 分离格式**: blocks 和 scales 存储在相邻但独立的内存区域，JIT 双指针读取，不需要预先合并（参见 `gllm/SPEC/07-LOADER.md §2.5 ARCH-MXFP4-SEPARATE`）。

### 5.6 AttentionVariant

| 变体 | 硬件 | 机制 | KV Cache |
|------|------|------|----------|
| `ScalarTiled { tile_q, tile_k }` | 任意 (fallback) | Cache-blocking 分块 | 无特殊要求 |
| `Avx2Tiled { tile_q, tile_k }` | x86 AVX2 | SIMD 向量化分块 | 连续内存 |
| `Avx512Tiled { tile_q, tile_k }` | x86 AVX-512 | zmm 向量化分块 | 连续内存 |
| `FlashV1 { block_q, block_k }` | GPU SM80+ | Shared memory tiling | Paged |
| `FlashV2 { block_q, block_k }` | GPU SM90+ | TMA + swizzle | Paged |
| `PagedTiled { block_q, page_size }` | CPU | Page-aware cache-blocking | PagedAttention |

### 5.7 NormVariant

| 变体 | 硬件 | 机制 |
|------|------|------|
| `ScalarLoop` | 任意 (fallback) | 两遍扫描: 平方和 → 缩放 |
| `SimdVectorized { width }` | AVX2/AVX-512/NEON/SVE | 向量化 reduce + scale, width 由 DeviceProfile 驱动 |
| `FusedNormGemm` | 任意 | Norm 输出直接喂入 GEMM (NormIntoGemm 融合模式) |

### 5.8 MoeVariant

| 变体 | 硬件 | 机制 |
|------|------|------|
| `ScalarDispatch` | CPU | 逐 token 逐专家循环 |
| `GpuBlockRouting { num_experts, top_k }` | GPU | Thread Block 级路由 + 汇编 jmp 跳转 |
| `FusedDequantExpert { quant_format }` | CPU | 融合 dequant + expert FFN (compute-time dequant) |

### 5.9 ElementwiseVariant

| 变体 | 硬件 | 机制 |
|------|------|------|
| `SimdLoop { width }` | 任意 | 向量化循环，width 由 `DeviceProfile.simd_width()` 决定 |

### 5.10 GatherVariant

| 变体 | 硬件 | 机制 |
|------|------|------|
| `ScalarLookup` | 任意 (fallback) | 标量索引循环: 读 index → 计算偏移 → 加载行 |
| `SimdGather { width }` | AVX2/AVX-512 | `vgatherdps` 向量化 gather |

### 5.11 策略选择算法

策略选择基于三路匹配: **OpClass × DeviceProfile × QuantFormat**

```rust
fn select_gemm_strategy(
    hw: &DeviceProfile,
    quant_a: QuantType,
    quant_b: QuantType,
) -> GemmVariant {
    match (hw.gpu_arch(), quant_a, quant_b) {
        // ── GPU 硬件 dequant 路径 (最高优先级) ──
        (Gfx12, Q4_K | Q4_0, Q8_0 | Q8_K) if hw.has_wmma_a4w8()
            => Rdna4WmmaA4W8,
        (Sm90, FP8, FP8) if hw.has_fp8_tensor_cores()
            => Sm90Fp8TensorCore,
        (Sm80, INT8 | UINT8, INT8 | UINT8) if hw.has_int8_tensor_cores()
            => Sm80Int8TensorCore,
        (Gfx942, INT8, INT8) if hw.has_mfma_int8()
            => Cdna3MfmaInt8,
        (Gfx942, FP8, FP8) if hw.has_mfma_fp8()
            => Cdna3MfmaFp8,

        // ── GPU 标准 tensor core / MFMA 路径 ──
        (Sm100, BF16, BF16) => Sm100Tcgen05 { .. },
        (Sm90, BF16 | F16, BF16 | F16)
            => Sm90WgmmaTma { num_waves: 2, cluster_dims: (1,1,1) },
        (Sm80, BF16 | F16, BF16 | F16)
            => Sm80MmaAsync { stages: 3, split_k: 1 },
        (Sm70, F16, F16) => Sm70Wmma { stages: 2 },
        (Gfx12, F16, F16) => Rdna4WmmaF16 { .. },
        (Gfx12, BF16, BF16) => Rdna4WmmaBf16 { .. },
        (Gfx11, F16, F16) => Rdna3WmmaF16 { .. },
        (Gfx11, BF16, BF16) => Rdna3WmmaBf16 { .. },
        (Gfx942, BF16, BF16) => Cdna3MfmaBf16 { .. },
        (Gfx942, F16, F16) => Cdna3MfmaF16 { .. },
        (Gfx90A, BF16, BF16) => Cdna2MfmaBf16 { .. },
        (Gfx90A, F16, F16) => Cdna2MfmaF16 { .. },

        // ── Apple GPU 路径 ──
        (AppleGpu, BF16, BF16) if hw.apple_gpu_gen() >= 9
            => MetalSimdMmBf16 { tile_m: 64, tile_n: 64 },
        (AppleGpu, F16, F16) if hw.apple_gpu_gen() >= 8
            => MetalSimdMM { tile_m: 64, tile_n: 64 },
        (AppleGpu, F32, F32) if hw.apple_gpu_gen() >= 7
            => MetalTiled { tile_m: 32, tile_n: 32 },

        // ── CPU fused dequant 路径 ──
        (_, Q4_K | Q4_0, F32) => FusedQ4Gemm { block_size: quant_a.block_size() },
        (_, Q8_0 | Q8_K, F32) => FusedQ8Gemm { block_size: quant_a.block_size() },
        (_, MXFP4, F32) => FusedMxfp4Gemm { block_size: 32 },

        // ── CPU 标准路径 ──
        (_, F32, F32) if hw.has_amx()        => AmxTile { tile_m: 16, tile_n: 16 },
        (_, BF16, F32) if hw.has_avx512_bf16() => Avx512Bf16 { mr: 16, nr: 6 },
        (_, F32, F32) if hw.has_avx512()      => Avx512F32 { mr: 16, nr: 6 },
        (_, F32, F32) if hw.has_avx2()        => Avx2Fma { mr: 6, nr: 4 },
        (_, F32, F32) if hw.has_neon()        => NeonFmla { mr: 8, nr: 4 },
        (_, F32, F32) if hw.has_sve()         => SveFmla { mr: hw.sve_vl_bytes() / 4, nr: 4 },

        // ── Fallback ──
        _ => ScalarLoop,
    }
}
```

### 5.12 策略属性

| 属性 | 说明 |
|------|------|
| **可缓存** | 同模型 + 同硬件 → 命中 `ImplementationPlan` 缓存，跳过策略选择 |
| **可审计** | 每个策略携带 `rationale: String`，日志输出 "选了哪个策略、为什么" |
| **可测试** | 单元测试 `select_strategy()` 不需要跑 E2E，不需要生成机器码 |
| **可比较** | A/B 测试不同策略变体，评估性能差异 |
| **可扩展** | 新增硬件变体 = 加一个 enum variant + 对应 codegen 分支 + 策略 match arm |

### 5.13 与 Phase 3 的交互

Phase 3 ISA Codegen 根据 `ImplementationPlan` 中每个 `GroupStrategy.compute` 发射机器码。Phase 3 是"哑翻译器"——不内嵌策略决策逻辑，只根据 `ComputeStrategy` 枚举值发射对应的指令序列:

```rust
// Phase 3 codegen 入口
fn emit_group(group: &GroupStrategy, emitter: &mut impl MachineCodeEmitter) {
    match &group.compute {
        ComputeStrategy::Gemm(Rdna4WmmaA4W8 { .. }) => emit_rdna4_wmma_a4w8(emitter),
        ComputeStrategy::Gemm(Avx2Fma { mr, nr }) => emit_avx2_fma_gemm(emitter, *mr, *nr),
        ComputeStrategy::DequantCompute(FusedQ4Gemm { block_size }) => emit_fused_q4_gemm(emitter, *block_size),
        ComputeStrategy::Attention(ScalarTiled { tile_q, tile_k }) => emit_tiled_attention(emitter, *tile_q, *tile_k),
        ComputeStrategy::Norm(SimdVectorized { width }) => emit_simd_norm(emitter, *width),
        // ... 所有变体
    }
}
```

**收益**: 策略选择与代码生成解耦。修改策略逻辑不影响 codegen，修改 codegen 不影响策略选择。

## 6. Phase 3: ISA Lowering

### 6.1 自动指令选择器 (Auto Instruction Selection)

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

#### 5.1.2 自动分发规则 — 三层分类体系

`emit_standalone_op` 必须按 `ComputePattern`（或等价的 `OpSemantics`）分类分发，不允许逐个 `OpKind` 手写 match arm。

**三层分类**:

| 层级 | ComputePattern | 分发策略 | 自动化级别 |
|------|---------------|---------|-----------|
| **Tier 1: 全自动** | Elementwise / BinaryElementwise | `auto_lower_trace(body)` → VmInstr | **100% 自动** — 零手写代码 |
| **Tier 1: 全自动** | Injective { body, num_inputs, num_outputs } | `emit_injective_inline` + `auto_lower_trace_multi` | **100% 自动** — trace 驱动，多输入多输出 |
| **Tier 1: 全自动** | Reduction { identity, combine, normalize } | `emit_reduction_inline` + `auto_lower_trace` | **100% 自动** — trace 驱动，归约+归一化 |
| **Tier 1: 全自动** | NormLike { reduce, finalize, transform } | `emit_normlike_inline` + `auto_lower_trace` / `auto_lower_trace_multi` | **100% 自动** — trace 驱动 |
| **Tier 2: 骨架手写 + 算术自动** | Gemm | `emit_gemm_inline_with_hook` | 分块策略手写，内层 FMA via TraceOp |
| **Tier 2: 骨架手写 + 算术自动** | TiledAttention | `emit_tiled_attention_inline` | 循环骨架手写，softmax/V-accum via TraceOp |
| **Tier 2: 骨架手写 + 算术自动** | MoEDispatch | `emit_moe_packed_inline` | 5-phase 骨架手写，router/softmax/topk/SwiGLU via TraceOp |
| **Tier 2: 骨架手写 + 算术自动** | QuantGemm | `emit_quant_gemm_inline` | 分块策略手写，dequant/FMA via TraceOp |
| **Tier 2: 骨架手写 + 算术自动** | Gather / ColumnSlice | `emit_*_inline` | 索引偏移手写，数据搬运 via TraceOp |
| **Tier 3: 纯控制流** | Structural (StoreToken/WriteLogits/Guardrail/...) | `dispatch_structural` | **100% 手写**（无算术） |

##### 5.1.2.1 "控制流骨架" 与 "算术计算体" 精确定义

**控制流骨架（允许手写）**:
- `emit_loop(BoundExpr, step, |prog, ctr, off|)` — 循环结构
- `VmInstr::ConditionalSkip` — 条件跳过（如 causal mask）
- `VmInstr::IndirectJump` — 间接跳转（如 hook 分发）
- `VmInstr::ScopeBegin / ScopeEnd` — 作用域管理
- `alloc_vreg(...)` — 虚拟寄存器分配
- `VmInstr::VecLoad / VecStore` — 纯数据加载/存储（无计算）
- `VmInstr::Broadcast { src: ScalarExpr::Param(...) }` — 参数广播

**算术计算体（禁止手写，必须通过 `auto_lower_trace_raw`）**:
- 所有 `VecBinOp { Add/Sub/Mul/Div/Max/Min }` — 算术运算
- 所有 `VecUnaryOp { Neg/Abs/Sqrt/Rsqrt/Recip }` — 一元运算
- 所有 `Transcendental { Exp/Log/Tanh/Sigmoid }` — 超越函数
- 所有 `Fma { acc, a, b }` — 融合乘加
- 所有 `VecCmp { pred }` — 比较
- 所有 `ConditionalSelect` — 条件选择
- 所有 `HReduce` — 水平归约
- 所有 `BroadcastScalar` + 后续运算 — 标量广播+计算
- 所有 `Mxfp4VecDequant` — 量化解量化

**核心原则**: 控制流骨架 = "决定在哪里计算"，算术计算体 = "计算什么"。后者必须是算法生成的。

**关键约束**：每个 ComputePattern 变体对应**一个**通用处理器函数，同类型的所有 OpKind 共享该处理器。禁止为特定 OpKind 创建 `emit_xxx_auto` 或 `lower_xxx` 等独立函数。

##### 5.1.2.2 ARCH-AUTO-INSTR-FULL 需求

```
REQ-AIS-FULL-001: 所有 Tier 2 算子的算术计算体必须通过 auto_lower_trace_raw 生成，
                  禁止在循环骨架内直接 emit(VmInstr::VecBinOp/VecUnaryOp/Transcendental/Fma/VecCmp/...)。
                  验证方式: grep emit_*_inline 函数中的 prog.emit(VmInstr::VecBinOp) 等，
                  结果必须为 0 或仅存在于 auto_select.rs 内部。

REQ-AIS-FULL-002: emit_loop 支持 Result 返回值（emit_loop_try），
                  使 auto_lower_trace_raw 可以在循环体内安全调用并传播错误。

REQ-AIS-FULL-003: TraceOp 枚举必须包含 Sigmoid 变体，
                  SwiGLU (x * sigmoid(gate)) 的 sigmoid 通过 auto_lower_trace_raw 生成，
                  禁止直接 emit(VmInstr::Transcendental { Sigmoid })。
```

#### 5.1.3 自动分发实现要求

1. **函数入口**: `emit_standalone_op` 函数顶部首先尝试 `try_auto_dispatch_by_pattern()`
2. **ComputePattern 路由**: 按 ComputePattern 类型路由到通用处理器（`emit_normlike_inline` / `emit_reduction_inline` / `emit_injective_inline` / `emit_gemm_inline_with_hook`）
3. **禁止手写 OpKind match**: 不允许 `OpKind::Xxx => { ... }` 形式的逐算子分发（仅 Opaque 类 `dispatch_structural` 允许按语义特征分组）
4. **禁止 per-OpKind 手写函数**: 不允许创建 `emit_layernorm_auto` / `emit_meanpool_auto` / `emit_rope_auto` / `lower_layernorm` / `lower_rope_full` / `lower_meanpool` 等按 OpKind 命名的手写 VmInstr 发射函数。所有算术表达式必须通过 `auto_lower_trace` / `auto_lower_trace_multi` 自动生成。现有 `emit_layernorm_auto` 为过渡实现，后续应并入 `emit_normlike_inline`
5. **新增算子**: 注册 scalar impl → SymExec 提取 trace → ComputePattern 自动分类 → 自动路由到对应通用处理器。无需编写任何额外代码
6. **缺失算子**: 走到 catch-all 时返回 `Err`，禁止静默 NOP（NO_SILENT_FALLBACK）

#### 5.1.4 TraceOp → VmInstr 自动查表

`auto_lower_trace()` 实现完全自动的 TraceOp → VmInstr 映射（`codegen/vm/auto_select.rs`）。

每个 TraceOp 变体自带映射语义，通过辅助函数消除重复代码：

**已实现 (auto_lower_trace dispatch_trace_op)**:

| TraceOp | VmInstr | 辅助函数 |
|---------|---------|---------|
| Input(n) | 直接返回 inputs[n] | — |
| Const(val) | Broadcast { ScalarExpr::Const(val) } | — |
| Add(a,b) / Sub / Mul / Div / Max / Min | VecBinOp { op } | `emit_binop()` — 6 变体共享 |
| Neg / Abs / Sqrt / Rsqrt / Recip | VecUnaryOp { op } | `emit_unary()` — 5 变体共享 |
| Fma(a,b,c) | Fma { acc, a, b } | — |
| Exp / Tanh / Log | Transcendental { func } | `emit_transcendental()` — 3 变体共享 |
| Compare { a, b, op } | VecCmp { pred } | `emit_cmp()` — CmpOp→CmpPredicate 映射 |
| Cast { src, from, to } | VecCast { from_bits, to_bits } | `quant_precision_bits()` — 13 种精度格式 |
| HReduce { src, op } | HReduce { op } | — 支持 Sum/Max/Min/Prod |

**未实现（auto_lower_trace 返回 Err）**:

| TraceOp | 需要的 VmInstr | 解锁能力 | 说明 |
|---------|---------------|---------|------|
| ConditionalBranch | VecConditional | 条件选择 (mask ? a : b) | 当前 Silu/Sigmoid 通过 Exp+Add+Div 替代 |
| QuantFma | 专用 path via StrategySelector | 混合精度 GEMM | TurboQuant 专用，不走 auto_lower_trace |
| BlockScale | 专用 path | 块缩放 (CDNA4) | 走 StrategySelector 专用 lower |
| Prefetch | PrefetchInstr | 质心预取 | 内存提示指令 |
| NonTemporalStore | NonTemporalStoreInstr | KV 连续写入绕缓存 | 内存提示指令 |
| BitExtract | BitExtractInstr | 量化解包 (nibbles→f32) | shift+mask / ubfx / bfe |
| Permute | VecPermute | 向量重排 (shuffle) | vpshufb / tbl / prmt |
| MaskedOp | MaskedVmInstr | 掩码操作 | 条件 lane 操作 |
| AtomicAdd | AtomicAddU32 (已存在) | MoE 命中计数 | lock xadd / ldadd / atomicAdd |
| FWHT | butterfly 序列 | TurboQuant FWHT | 展开的加减指令序列 |

**关键保证**: 新增 TraceOp 变体只需在 `auto_lower_trace` 的 match 中添加**一行**映射，不需要修改任何其他 lower 函数。

#### 5.1.5 TraceOp 扩展路线图

当前 17 种 TraceOp 中，10 种已在 `auto_lower_trace` 中实现（基础算术 + Compare + Cast + HReduce）。剩余 7 种走专用路径或待实现：

| 类别 | TraceOp | 路径 | 依赖 |
|------|---------|------|------|
| **控制流** | ConditionalBranch | auto_lower_trace + VecConditional | 新增 VmInstr::VecConditional |
| **量化专用** | QuantFma / BlockScale | StrategySelector 专用 lower | 不走 auto_lower_trace |
| **内存优化** | Prefetch / NonTemporalStore | auto_lower_trace + 内存提示 | 新增对应 VmInstr |
| **位操作** | BitExtract | auto_lower_trace | 量化解包场景 |
| **向量操作** | Permute | auto_lower_trace | 向量重排场景 |
| **掩码** | MaskedOp | auto_lower_trace | 条件 lane 操作 |
| **原子** | AtomicAdd | auto_lower_trace (VmInstr 已存在) | MoE 命中计数 |
| **信号处理** | FWHT | auto_lower_trace (butterfly 展开) | TurboQuant 专用 |

**ConditionalBranch 替代路径**: 当前 Silu/Sigmoid/Gelu 等 trace 不使用 ConditionalBranch，而是通过 Exp+Add+Div 算术等价实现。这使得这些算子的全自动 lowering 不依赖 ConditionalBranch 的实现。

### 5.1.7 Category D 消除 (REQ-AIS-005)

`emit_standalone_op` 当前仍存在 ~13 个手写 `OpKind::Xxx =>` match arm（Category D）。REQ-AIS-005 要求全部消除，改为 ComputePattern 驱动路由。

**消除策略矩阵**:

| OpKind | 消除策略 | 目标路径 | 依赖 |
|--------|---------|---------|------|
| Silu | auto elementwise | 已有 scalar 注册，SymExec trace 含 Exp+Add+Div | 无 |
| Residual | auto elementwise (BinaryElementwise) | Add 操作，已有 scalar 注册 | 无 |
| LogitSoftcap | auto elementwise | tanh + mul，已有 scalar 注册 | 无 |
| Argmax | Reduction pattern | VmInstr::Argmax 已存在 | 无 |
| StoreToken | structural VecStore | 带 output buffer 偏移计算的 VecStore | 无 |
| WriteLogits | structural VecStore | 带 logits buffer 偏移的 VecStore | 无 |
| CheckStopCondition | structural 控制流 | 比较+条件跳转 VmInstr | 无 |
| EarlyExit | structural 控制流 | ConditionalSkip/ConditionalExit VmInstr 已存在 | 无 |
| GuardrailCheck | structural 控制流 | 比较共享内存 veto 标志 + ConditionalSkip | 无 |
| CotStepCheck | structural 共享内存 | 读共享内存 step counter + 条件跳转 | 无 |
| SgInject | structural 共享内存 | 写共享内存 ring buffer | 无 |
| SgDetect | structural 共享内存 | 读共享内存 ring buffer + 计算 | 无 |
| Reshape/Transpose/SliceView | NOP | 纯元数据操作 | 无 |

**目标架构**:

```
emit_standalone_op(op_id)
  │
  ├─ 1. try_auto_dispatch_by_pattern()
  │    ├─ Elementwise / BinaryElementwise
  │    │    → auto_lower_trace() 自动完成
  │    │    覆盖: Silu, Residual, LogitSoftcap, 所有激活函数
  │    │
  │    ├─ Injective { body, num_inputs, num_outputs }
  │    │    → emit_injective_inline + auto_lower_trace_multi
  │    │    覆盖: RoPE, 所有多输入多输出逐元素变换
  │    │
  │    └─ Reduction { identity, combine, normalize }
  │         → emit_reduction_inline + auto_lower_trace
  │         覆盖: MeanPool, L2Normalize, Argmax, 所有归约算子
  │
  ├─ 2. dispatch_compute_pattern()
  │    ComputePattern::NormLike → emit_normlike_inline
  │    ComputePattern::Gemm → emit_gemm_inline_with_hook
  │    ComputePattern::QuantDecode → lower_quant_decode
  │    (路由键是 ComputePattern，不是 OpKind)
  │
  └─ 3. structural_ops()
       Reshape/Transpose/SliceView → NOP
       StoreToken/WriteLogits → VecStore (带偏移)
       CheckStopCondition/EarlyExit/GuardrailCheck → 控制流 VmInstr
       SgInject/SgDetect/CotStepCheck → 共享内存 VmInstr
       未匹配 → Err (NO_SILENT_FALLBACK)
```

**关键约束**:
- 每个 ComputePattern 变体对应一个**通用处理器**（不是 per-OpKind 函数）
- 禁止创建 `emit_layernorm_auto` / `emit_meanpool_auto` / `emit_rope_auto` / `lower_layernorm` 等 per-OpKind 手写函数
- 新增 OpKind 只需注册 scalar impl → SymExec 提取 ComputePattern → 自动路由到对应通用处理器
- `structural_ops` 按 OpKind 语义特征（内存/控制流/元数据）分发，非逐个 OpKind 手写

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
| Compare(a,b,op) | vcmpps{k} → k-mask (AVX-512) / vcmpps + movemask (AVX2) |
| Cast(a, F16→F32) | vcvtph2ps (F16C) |
| Cast(a, BF16→F32) | vdpbf16ps 输入侧 (AVX-512 BF16) |
| HReduce(a, Sum) | vhaddps + vperm + vadd (4 级) |
| HReduce(a, Max) | vmaxps + vperm + vcmp + blend (4 级) |

**AArch64 (dynasm-rs)**:

| TraceOp | NEON (v.4s) / SVE |
|---------|-------------------|
| Add(a,b) | fadd |
| Fma(a,b,c) | fmla |
| Exp(a) | 多项式逼近 |
| Sqrt(a) | fsqrt |
| Compare(a,b,op) | fcmp → fcsel / SVE whilelo + fcm (predicate) |
| Cast(a, F16→F32) | fcvtl |
| Cast(a, BF16→F32) | bfdot 输入侧 / bfmmla |
| HReduce(a, Sum) | faddp (pairwise) + addv (SVE) |

**GPU (PTX)**:

| TraceOp | PTX |
|---------|-----|
| Add | add.f32 |
| Fma | fma.rn.f32 |
| Exp | ex2.approx.f32 + 多项式修正 |
| Compare(a,b,op) | setp.{eq/ne/lt/le/gt/ge}.f32 → predicate reg |
| Cast(a, F16→F32) | cvt.f32.f16 |
| Cast(a, BF16→F32) | cvt.f32.bf16 |
| HReduce(a, Sum) | shfl.sync ↓ + add (warp reduce) |
| HReduce(a, Max) | shfl.sync ↓ + max (warp reduce) |

### 6.2 PlatformBackend 统一接口

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

### 6.3 禁止事项

- 禁止跳过 Lifting 阶段直接手写 ISA 汇编（如直接写 AVX2 Softmax）
- 禁止在 JIT 代码中通过 `call` 指令调用预编译的 Rust/C 函数
- 禁止写死特定 ISA，必须通过 DeviceProfile 参数化
- 禁止 emit_nop() 作为未实现 op 的 catch-all（见 `00-PHILOSOPHY.md` NO_SILENT_FALLBACK）
- 硬件差异体现在 codegen 层的指令选择，不是 fusion 层的算子拆分（见 `00-PHILOSOPHY.md` NO_HW_DEGRADATION）

## 7. JIT 缓存协议

### 7.1 编译时机

- 编译只发生在模型加载时
- 推理热路径（decode step 层循环）中禁止任何编译行为
- `InferenceCompiler::new()` / `compile_graph()` / `build_*_graph()` 只在模型加载时调用

### 7.2 SymDim 动态维度

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

### 7.3 三级缓存

| 级别 | 作用域 | 存储位置 | 生命周期 |
|------|--------|---------|---------|
| L1 | 模型级 | 内存 | 模型卸载时释放 |
| L2 | 全局 LRU | 内存 | 进程退出时释放 |
| L3 | 持久化 | `~/.gllm/jit_cache/` | 7 天 TTL 自动清理 |

Debug 模式（`cargo test` / `cargo build`）: L3 磁盘缓存完全禁用。
Release 模式（`cargo build --release`）: L3 正常工作。

### 7.4 CPU/GPU 统一 (ARCH-CPU-GPU-UNIFIED)

CPU 和 GPU 后端共享完全一致的 `CompilerGraph` IR 和 `GraphType`。

- 禁止以 `Cpu`/`Gpu`/后端名称为前缀的 `GraphType` 变体（如 `CpuDecoderLayer`）
- 禁止 CPU 专用图构建器（如 `build_decoder_layer_graph()`）
- 禁止子算子级 `GraphType`（如 `KvProjection`、`QRope`）
- CPU 和 GPU 给定相同输入和权重，输出必须在浮点精度范围内一致

### 7.5 编译粒度

缓存粒度 = 全层融合图：

- `FusedAttentionLayer`（RmsNorm + Q/K/V GEMM + RoPE + Attention + O GEMM）
- `FusedFfnLayer`（RmsNorm + Gate/Up GEMM + Activation + Down GEMM）

禁止子算子级缓存（如单独缓存一个 RmsNorm 或一个 GEMM）。

## 8. CompilerGraph 接口边界

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
