# PIPELINE-CODEGEN: JIT Code Generation Pipeline Specification

## 1. Overview

JIT 代码生成管线将 Phase 2 融合决策 (`FusionPlan`) 翻译为 Phase 3 可执行机器码。
管线以 **Pipeline Stage** 为基本执行单位，通过 **StageContext** 传递融合约束、
布局信息和后端能力，确保每个 Stage 的输入/输出/寄存器/偏移都是可验证的计算属性。

### 1.1 设计目标

| 目标 | 约束 |
|------|------|
| 零硬编码 | 所有 byte offset、buffer size、寄存器分配必须从 Layout + BackendCapabilities 推导 |
| 融合可组合 | 任意两个 Pipeline Stage 可以通过 StageConnector 串联/嵌套 |
| 后端无关 | Pipeline Stage 只通过 BackendEmitter trait 操作，不接触物理指令 |
| 形式化验证 | 每个 Stage 的前置/后置条件可在编译时检查 |

### 1.2 管线全景

```
FusionPlan + CompilerGraph + BufferAllocation
         │
         ▼
  ┌──────────────────────────────────┐
  │  Stage 0: Plan Translation       │
  │  FusionGroup → PipelinePlan      │
  │  (code_generator_bridge.rs)      │
  └──────────┬───────────────────────┘
             │ PipelinePlan { stages: Vec<PipelineStage> }
             ▼
  ┌──────────────────────────────────┐
  │  Stage 1: Layout Derivation      │
  │  PipelineStage → StageContext    │
  │  (layout.rs + hw_constraints)    │
  └──────────┬───────────────────────┘
             │ StageContext { layout, constraints, ptr_bindings }
             ▼
  ┌──────────────────────────────────┐
  │  Stage 2: Algorithm Emission     │
  │  StageContext → CodeGenerator    │
  │  (algorithm_driver.rs)           │
  └──────────┬───────────────────────┘
             │ CodeGenerator<B> 已发射指令
             ▼
  ┌──────────────────────────────────┐
  │  Stage 3: Finalize               │
  │  Epilogue + ConstPool + Assemble │
  │  (x86_backend.rs / aarch64 / …) │
  └──────────┬───────────────────────┘
             │ CodegenOutput { code, scratchpad_bytes }
             ▼
         可执行字节码
```

## 2. Core Types

### 2.1 PipelineStage — 单个代码生成阶段

```rust
/// 管线中的一个代码生成阶段。
/// 每个 PipelineStage 对应一个 FusionGroup 的代码生成任务。
pub struct PipelineStage {
    /// 融合模式（决定选择哪个 Algorithm Driver）
    pub kind: StageKind,
    /// 指针绑定配置（从 graph + alloc + weight_layout 推导）
    pub ptr_bindings: PtrBindings,
    /// 布局约束（从 OpKind + BackendCapabilities 推导）
    pub layout: StageLayout,
    /// 要执行的 TraceOp 序列（从 ScalarOpRegistry 提取）
    pub body: Vec<TraceOp>,
    /// 可选的 epilogue TraceOps（融合的后续算子）
    pub epilogue: Option<Vec<TraceOp>>,
}
```

### 2.2 StageKind — 阶段类型枚举

```rust
pub enum StageKind {
    /// 逐元素 SIMD 循环 (LoopFusion / Standalone elementwise)
    Elementwise,
    /// BLIS GEMM (Standalone / EpilogueInjection)
    Gemm,
    /// Norm 三阶段 (Reduce → Finalize → Transform)
    Norm,
    /// RoPE 旋转位置编码
    Rope,
    /// MHA 多头注意力 (Score → Softmax → ValueAggregation)
    Attention,
    /// MeanPool 列平均
    MeanPool,
    /// 复合融合 (NormIntoGemm / TileLevelFusion / QkvSharedInput / FFNBlock)
    Compound(CompoundKind),
}

pub enum CompoundKind {
    /// Norm → GEMM (scratchpad 驻留)
    NormThenGemm,
    /// GEMM MC 循环内嵌入前驱 (TileLevelFusion)
    TileFused { mc: usize },
    /// 三次 GEMM 共享 pack_a (QkvSharedInput)
    QkvShared,
    /// Gate+Up GEMM → Activation → Mul → Down GEMM (FFNBlock)
    FfnBlock,
    /// Residual Add + 下层 Norm (CrossLayerResidual)
    CrossLayerResidual,
    /// QKV + QkNorm + ValueNorm + RoPE (Mega-Kernel §9)
    MegaQkvNormRope,
}
```

### 2.3 StageLayout — 阶段布局描述

```rust
/// 阶段的完整布局——所有维度、步幅、buffer size 都从此推导。
/// Algorithm Driver 禁止硬编码任何维度值。
pub enum StageLayout {
    /// 逐元素: elem_count + SIMD 对齐
    Elementwise(ElementwiseParams),
    /// GEMM: m × n × k + BLIS blocking
    Gemm(GemmParams),
    /// Norm: feature_dim + 三阶段 pattern
    Norm(NormParams),
    /// RoPE: num_heads × head_dim × half
    Rope(RopeParams),
    /// MHA: seq × heads × kv_heads × head_dim
    Attention(AttentionParams),
}

pub struct ElementwiseParams {
    /// 总元素数量
    pub elem_count: usize,
    /// SIMD 向量数量 = elem_count / lanes
    pub vec_count: usize,
    /// 标量尾部元素数量 = elem_count % lanes
    pub tail: usize,
    /// 每次迭代的字节步进 = lanes * elem_bytes
    pub step_bytes: usize,
    /// 是否是二元算子
    pub is_binary: bool,
}

pub struct GemmParams {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// BLIS blocking 参数（从 BackendCapabilities 推导）
    pub blocking: BlisBlocking,
    /// pack_a scratchpad size
    pub pack_a_bytes: usize,
    /// pack_b scratchpad size
    pub pack_b_bytes: usize,
}

pub struct BlisBlocking {
    pub mr: usize,   // 微内核行数
    pub nr: usize,   // 微内核列数
    pub mc: usize,   // IC 循环 tile (L1)
    pub nc: usize,   // JC 循环 tile (L3)
    pub kc: usize,   // PC 循环 tile (L2)
}

pub struct NormParams {
    pub feature_dim: usize,
    pub vec_count: usize,
    pub tail: usize,
    pub row_bytes: usize,
    pub pattern: ComputePattern,  // NormLike { reduce, finalize, transform }
}

pub struct RopeParams {
    pub num_heads: usize,
    pub head_dim: usize,
    pub half: usize,
    pub rope_base: f32,
    pub max_seq_len: usize,
}

pub struct AttentionParams {
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,  // 1/sqrt(head_dim)
    pub scratchpad_bytes: usize,
}
```

### 2.4 PtrBindings — 指针绑定配置

```rust
/// 每个 Stage 的指针绑定——从 graph + alloc + weight_layout 推导。
/// Algorithm Driver 通过 VReg::Ptr(kind) 访问，不关心物理来源。
pub struct PtrBindings {
    pub activation: PtrSource,        // VReg::Ptr(PtrKind::Activation)
    pub weight: Option<PtrSource>,    // VReg::Ptr(PtrKind::Weight)
    pub output: PtrSource,            // VReg::Ptr(PtrKind::Output)
    pub scratchpad: Option<PtrSource>,// VReg::Ptr(PtrKind::Scratchpad)
    pub positions: Option<PtrSource>, // VReg::Ptr(PtrKind::Positions)
}
```

## 3. LoopContext 扩展 — Byte Offset 自动步进

### 3.1 问题

当前 `emit_loop` 的回调无法获得循环内的当前 byte offset：

```rust
// 当前（错误）
gen.emit_loop(LoopBound::Const(vec_count), |gen| {
    let offset = 0; // ❌ 硬编码！
    gen.vec_load(VReg::Acc(0), VReg::Ptr(PtrKind::Activation), offset)?;
    ...
});
```

### 3.2 解决方案: StrideLoop

新增 `emit_stride_loop` 方法，自动维护 byte offset 步进：

```rust
impl<B: BackendEmitter> CodeGenerator<B> {
    /// 发射带步进的循环。
    ///
    /// 每次迭代自动步进 `step_bytes`。回调接收 `LoopIter` 提供:
    /// - `byte_offset()` — 当前迭代的字节偏移 (counter * step_bytes)
    /// - `iteration()` — 当前迭代索引
    ///
    /// 后端实现: counter register 乘以 step_bytes → 存入临时 GPR。
    pub fn emit_stride_loop(
        &mut self,
        count: usize,
        step_bytes: usize,
        body: impl FnOnce(&mut Self, &LoopIter) -> Result<(), CompilerError>,
    ) -> Result<(), CompilerError>;
}

/// 循环迭代上下文——由 emit_stride_loop 创建，传给 body 回调。
pub struct LoopIter {
    /// 当前迭代的字节偏移 = counter * step_bytes
    /// 后端将此映射到 `counter_reg * step_bytes` 指令序列。
    byte_offset: VReg,  // 持有 byte offset 的 GPR
    step_bytes: usize,
}

impl LoopIter {
    /// 基于当前 byte offset 计算加载地址偏移
    pub fn load_offset(&self, base_offset: usize) -> usize {
        // 编译时常量 base + 运行时 counter*step
        base_offset  // 运行时部分由后端通过 byte_offset VReg 处理
    }
}
```

### 3.3 Algorithm Driver 使用示例

```rust
pub fn emit_elementwise<B: BackendEmitter>(
    gen: &mut CodeGenerator<B>,
    params: &ElementwiseParams,
    body: &[TraceOp],
) -> Result<(), CompilerError> {
    if params.vec_count > 0 {
        gen.emit_stride_loop(params.vec_count, params.step_bytes, |gen, iter| {
            // iter.byte_offset 自动 = counter * step_bytes
            gen.vec_load_strided(VReg::Acc(0), VReg::Ptr(PtrKind::Activation), iter)?;

            if params.is_binary {
                gen.vec_load_strided(VReg::Temp(0), VReg::Ptr(PtrKind::Weight), iter)?;
            }

            gen.emit_trace(body)?;

            gen.vec_store_strided(VReg::Ptr(PtrKind::Output), iter, VReg::Acc(0))?;
            Ok(())
        })?;
    }

    // Tail handling
    if params.tail > 0 {
        gen.emit_scalar_tail(params.tail, params.step_bytes * params.vec_count, |gen, elem_off| {
            // 逐元素标量处理
            gen.scalar_load(VReg::Acc(0), VReg::Ptr(PtrKind::Activation), elem_off)?;
            gen.emit_trace_with_width(body, SimdWidth::Scalar)?;
            gen.scalar_store(VReg::Ptr(PtrKind::Output), elem_off, VReg::Acc(0))?;
            Ok(())
        })?;
    }

    Ok(())
}
```

### 3.4 后端实现 (BackendEmitter 扩展)

```rust
pub trait BackendEmitter {
    // ... existing methods ...

    /// 发射带步进的循环头。
    /// 后端维护 counter_reg 和 byte_offset_reg = counter * step。
    fn emit_stride_loop_header(
        &mut self,
        counter: VReg,
        count: usize,
        step_bytes: usize,
    ) -> Result<(LoopHandle, VReg), CompilerError>;
    //                       ^-- byte_offset VReg

    /// 带步进的向量加载: base_ptr + byte_offset_vreg
    fn vec_load_strided(
        &mut self,
        dst: VReg,
        base: VReg,
        offset_vreg: VReg,
        width: SimdWidth,
    ) -> Result<(), CompilerError>;

    /// 带步进的向量存储: base_ptr + byte_offset_vreg
    fn vec_store_strided(
        &mut self,
        base: VReg,
        offset_vreg: VReg,
        src: VReg,
        width: SimdWidth,
    ) -> Result<(), CompilerError>;
}
```

X86 后端实现:
```
emit_stride_loop_header(counter=Counter(0), count=N, step=32):
  xor r12, r12           ; counter = 0
  xor rbx, rbx           ; byte_offset = 0
.loop_start:
  cmp r12, N
  jge .loop_done

emit_stride_loop_footer:
  inc r12                 ; counter++
  add rbx, 32            ; byte_offset += step
  jmp .loop_start
.loop_done:

vec_load_strided(dst=ymm0, base=rdi, offset=rbx):
  vmovups ymm0, [rdi + rbx]

vec_store_strided(base=r8, offset=rbx, src=ymm0):
  vmovups [r8 + rbx], ymm0
```

## 4. Fusion Pipeline Composition — 融合算子的 Pipeline 连接

### 4.1 设计原则

融合不是在 Algorithm Driver 层面做的——融合决策在 Phase 2 (FusionEngine) 完成。
Algorithm Driver 层的职责是：给定一个 `CompoundKind`，将多个 Stage 按正确顺序组合。

### 4.2 CompoundStage 执行模型

```rust
/// 复合 Stage: 将多个简单 Stage 按特定拓扑组合
pub enum CompoundExecution {
    /// 顺序执行: Stage A → Stage B (A 的输出 = B 的输入)
    /// 用于 NormThenGemm: Norm → GEMM
    Sequential(Vec<PipelineStage>),

    /// 嵌套执行: Stage A 的外循环内嵌入 Stage B
    /// 用于 TileLevelFusion: GEMM.MC_loop { Norm(mc_rows) → GEMM.micro_kernel }
    Nested {
        outer: PipelineStage,       // GEMM (控制 MC loop)
        inner: PipelineStage,       // Norm (每 MC strip 执行)
        nest_point: NestPoint,      // 嵌入点 = MC loop body start
    },

    /// 共享输入: 多个 Stage 共享同一 pack 结果
    /// 用于 QkvSharedInput: pack_a(input) → GEMM_Q / GEMM_K / GEMM_V
    SharedInput {
        pack_stage: PipelineStage,  // pack_a
        consumers: Vec<PipelineStage>,  // Q/K/V projections
    },

    /// 多阶段流水线: 按依赖顺序执行
    /// 用于 FFNBlock: [Gate GEMM, Up GEMM] → SiLU(gate) → Mul(gate, up) → Down GEMM
    MultiStage(Vec<PipelineStage>),
}

pub enum NestPoint {
    /// 嵌入到 GEMM 的 MC 循环起始处
    GemmMcLoopStart,
    /// 嵌入到 GEMM 的 PC 循环起始处 (pack_b 之后)
    GemmPcLoopAfterPackB,
}
```

### 4.3 Scratchpad 生命周期管理

```rust
/// Scratchpad 区域的生命周期描述。
/// 管线使用 RAII 风格分配——每个 Stage 声明需要的 buffer，
/// Pipeline 调度器保证同一时刻不超过 L1/L2/L3 budget。
pub struct ScratchpadRegion {
    pub offset: usize,
    pub size: usize,
    pub label: &'static str,
    /// 生命周期: 从哪个 Stage 开始有效，到哪个 Stage 释放
    pub producer: StageId,
    pub last_consumer: StageId,
    /// 缓存层级约束
    pub cache_level: CacheLevel,
}

pub enum CacheLevel {
    L1Resident,    // 必须驻留 L1 (TileLevelFusion 的中间结果)
    L2Resident,    // 可以在 L2 (pack_a, pack_b)
    L3OrMemory,    // 大 buffer (完整矩阵)
}
```

## 5. Epilogue Injection Pipeline — 累加器上的就地融合

### 5.1 Epilogue 在 GEMM Pipeline 中的位置

```
BLIS 5-Level Loop:
  JC → PC → pack_b → IC → pack_a → JR → IR → {
    // Micro-kernel: kc 次 FMA
    for pp in 0..kc:
      acc[r][c] += a[r, pp] * b[pp, c]

    // === Epilogue Injection Point ===
    // 在累加器寄存器上就地执行融合算子
    for each acc[r][c]:
      acc[r][c] = epilogue(acc[r][c])  // bias, activation, residual

    // Store
    C[i+ir, j+jr] = acc[r][c]
  }
```

### 5.2 Epilogue 可表达的操作

每个 Epilogue TraceOp 在累加器 ymm/zmm 上就地执行：

| 操作 | TraceOp | 寄存器行为 |
|------|---------|----------|
| Bias Add | `Add(acc, bias_vec)` | `vaddps ymm_acc, ymm_acc, [bias_ptr + col*4]` |
| SiLU | `sigmoid(x)*x` | Cephes exp polynomial on ymm_acc |
| GeLU | `0.5*x*(1+tanh(...))` | polynomial + tanh on ymm_acc |
| Residual | `Add(acc, residual)` | `vaddps ymm_acc, ymm_acc, [residual_ptr + offset]` |
| Scale | `Mul(acc, scale)` | `vmulps ymm_acc, ymm_acc, ymm_scale` |

### 5.3 Epilogue Pipeline 约束

```rust
pub struct EpilogueConfig {
    /// Epilogue TraceOp 序列
    pub ops: Vec<TraceOp>,
    /// 额外的 ptr 绑定 (bias, residual 等)
    pub extra_ptrs: Vec<(PtrKind, PtrSource)>,
    /// 寄存器压力: epilogue 需要的额外 temp 向量数
    pub temp_vectors_needed: usize,
}

// 约束检查:
//   temp_vectors_needed + mr*nr ≤ max_vectors
//   (不能让 epilogue 把累加器挤出寄存器)
```

## 6. ISA Specialization Pipeline — 后端特化注入点

### 6.1 特化点清单

Algorithm Driver 定义算法骨架，后端在特定点注入 ISA 优化：

| 特化点 | 触发条件 | AVX2 实现 | AVX-512 实现 | NEON 实现 |
|--------|---------|----------|-------------|----------|
| GEMM Pack | `tile_compute()` 有 AMX | ymm load/store | zmm + k-mask | v.4s load/store |
| FMA Body | `has_fma` | `vfmadd231ps` | `vfmadd231ps zmm` | `fmla v.4s` |
| Horizontal Sum | Reduction | `vhaddps` × 3 | `vextractf64x4` + vhadd | `faddp` + `faddp` |
| Exp/Tanh/Log | transcendental | Cephes poly | Cephes poly (wider) | NEON poly |
| BF16 GEMM | `has_native_bf16` | F32 fallback | `vdpbf16ps` | — |
| Tile GEMM | AMX/SME | — | `tdpbf16ps` | SME `smopa` |

### 6.2 BackendEmitter 扩展接口

```rust
pub trait BackendEmitter {
    // ... existing ...

    /// 后端推荐的 BLIS blocking 参数
    fn recommended_blocking(&self) -> BlisBlocking {
        let caps = self.capabilities();
        BlisBlocking::derive(caps)
    }

    /// 是否支持累加器上的就地 epilogue
    fn supports_epilogue_on_accumulators(&self) -> bool { true }

    /// Pack A/B 的最优实现（后端可覆盖提供 SIMD pack）
    fn emit_pack_a(&mut self, _src: VReg, _dst: VReg, _mc: usize, _kc: usize)
        -> Result<(), CompilerError> {
        Err(CompilerError::CodegenViolation("pack_a not implemented".into()))
    }

    fn emit_pack_b(&mut self, _src: VReg, _dst: VReg, _kc: usize, _nc: usize)
        -> Result<(), CompilerError> {
        Err(CompilerError::CodegenViolation("pack_b not implemented".into()))
    }
}
```

## 7. Pipeline Execution Protocol

### 7.1 codegen_plan 重构为 Pipeline 驱动

```rust
pub fn codegen_plan(
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    exec_plan: &ExecutionPlan,
    registry: Option<&ScalarOpRegistry>,
) -> Result<CodegenOutput, CompilerError> {
    let profile = &exec_plan.profile;
    let weight_layout = graph.weight_layout();

    // Stage 0: Plan Translation
    let pipeline = translate_plan(plan, graph, alloc, &weight_layout, registry)?;

    // Stage 1: Layout Derivation (enrich with backend capabilities)
    let caps = BackendCapabilities::from_profile(profile);
    let enriched = enrich_layouts(&pipeline, &caps)?;

    // Create backend + generator
    let mut backend = JitX86Backend::new(profile);
    backend.emit_prologue()?;
    setup_abi(&mut backend, &weight_layout)?;
    let mut gen = CodeGenerator::new(backend, alloc.total_bytes);

    // Stage 2: Algorithm Emission
    for stage in &enriched.stages {
        // Bind pointers
        bind_stage_pointers(&mut gen, &stage.ptr_bindings)?;

        // Emit algorithm
        match &stage.kind {
            StageKind::Elementwise => {
                let params = stage.layout.as_elementwise()?;
                emit_elementwise(&mut gen, params, &stage.body)?;
            }
            StageKind::Gemm => {
                let params = stage.layout.as_gemm()?;
                emit_gemm(&mut gen, params, stage.epilogue.as_deref())?;
            }
            StageKind::Norm => {
                let params = stage.layout.as_norm()?;
                emit_norm(&mut gen, params)?;
            }
            StageKind::Compound(kind) => {
                emit_compound(&mut gen, kind, stage)?;
            }
            // ...
        }
    }

    // Stage 3: Finalize
    gen.backend.emit_epilogue()?;
    let scratchpad_bytes = gen.state.scratchpad_watermark;
    let code = gen.finalize()?;

    Ok(CodegenOutput { code, scratchpad_bytes, hotpatch_points: vec![] })
}
```

### 7.2 Pipeline Validation

每个 Stage 执行前后，Pipeline 运行时检查：

```rust
pub struct StageValidation {
    /// 前置条件: 需要哪些 VReg 已绑定
    pub requires: Vec<(VReg, VRegBinding)>,
    /// 后置条件: 哪些 VReg 被 clobber
    pub clobbers: Vec<VReg>,
    /// 内存约束: scratchpad 使用不超过预算
    pub scratchpad_budget: usize,
    /// 寄存器约束: 同时活跃的向量寄存器不超过后端限制
    pub max_live_vectors: usize,
}
```

## 8. SPEC Optimization Algorithm Mapping

### 8.1 §9 Mega-Kernel Block Routing → CompoundKind::MegaQkvNormRope

```rust
// QKV + QkNorm + ValueNorm + RoPE 全融合
emit_compound(gen, CompoundKind::MegaQkvNormRope, stage):
  1. SharedInput: pack_a(input) → scratchpad
  2. GEMM_Q: pack_a × W_q → Q (scratchpad q_region)
  3. GEMM_K: pack_a × W_k → K (scratchpad k_region)
  4. GEMM_V: pack_a × W_v → V (output v_region)
  5. QkNorm: L2Normalize(Q), L2Normalize(K) (in-place scratchpad)
  6. RoPE: rotate(Q, cos_sin), rotate(K, cos_sin) (in-place scratchpad)
  7. Copy Q, K to output regions
```

### 8.2 §13 Epilogue Freeloading → EpilogueConfig

每个 GEMM Stage 可选附带 EpilogueConfig，在累加器上就地融合 11 种操作。

### 8.3 §11 TurboQuant → StageKind::Gemm + QuantPrecision

量化 GEMM 通过 GemmParams 的 precision 字段驱动后端选择反量化指令：
```rust
pub struct GemmParams {
    // ...
    pub precision: MixedPrecision,  // act=F16, weight=INT4, acc=F32
}
```

### 8.4 §16 Residual Bus → CompoundKind::CrossLayerResidual

跨层残差 + 下层 Norm 融合为单个 Compound Stage。

## 9. Implementation Priority

| Phase | 内容 | 解锁的 E2E 测试 |
|-------|------|----------------|
| P0 | StrideLoop + ElementwiseParams | SiLU, GeLU, Add, Mul |
| P1 | GEMM blocking 参数推导 + Pack A/B | gemm_single_tile, gemm_full |
| P2 | Norm 三阶段 + Row Loop | norm_row |
| P3 | EpilogueConfig on accumulators | gemm_with_epilogue |
| P4 | Compound: NormThenGemm, TileFused | norm_into_gemm |
| P5 | RoPE + MHA + CachedGQA | attention tests |
| P6 | QkvShared + MegaQkvNormRope | full model |
| P7 | Telemetry + HotPatch | observability |
