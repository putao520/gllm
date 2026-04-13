# JIT 代码生成器架构设计

> **SSOT**: 所有 ISV 后端的 JIT 代码生成器必须符合此架构。

## 1. 核心洞察

**OP 抽象已经存在——就是 `TraceOp`。**

标量函数 → Phase 0 符号执行 → `Vec<TraceOp>` = **后端无关的算子语义**。

```
scalar_silu(x) → SymExec → [Input(0), Neg(0), Exp(1), Const(1.0), Add(2,3), Recip(4), Mul(0,5)]
```

这 7 个 TraceOp 就是 SiLU 的本质——跟 x86/aarch64/PTX 无关。

每个后端拿到同一个 `Vec<TraceOp>`，各自生成最优指令：

| TraceOp | x86 AVX2 | x86 AVX-512 | aarch64 NEON | PTX |
|---------|----------|-------------|--------------|-----|
| `Add(a,b)` | `vaddps ymm` | `vaddps zmm` | `fadd v.4s` | `add.f32` |
| `Mul(a,b)` | `vmulps ymm` | `vmulps zmm` | `fmul v.4s` | `mul.f32` |
| `FMA(a,b,c)` | `vfmadd231ps ymm` | `vfmadd231ps zmm` | `fmla v.4s` | `fma.rn.f32` |
| `Exp(x)` | 多项式逼近 (12 指令) | 多项式逼近 (8 指令) | 多项式逼近 | `ex2.approx.f32` |
| `Rsqrt(x)` | `vrsqrtps + Newton` | `vrsqrt14ps` | `frsqrte + Newton` | `rsqrt.approx.f32` |

**不需要重新设计 OP。需要的是：一个有状态的生成器，控制 TraceOp → 物理指令的映射过程。**

## 2. 三层架构

```
┌─────────────────────────────────────────────────────┐
│  Layer 3: Algorithm Driver (后端无关)                │
│                                                     │
│  输入: CompilerGraph + FusionPlan + ComputePattern  │
│  控制: 循环结构、分块策略、融合模式                   │
│  输出: 对 Layer 2 的调用序列                         │
│                                                     │
│  · GEMM: NC→KC→MC→NR→microkernel→epilogue          │
│  · Elementwise: single-pass SIMD loop               │
│  · NormLike: reduce→finalize→transform              │
│  · Reduction: init→combine→finalize                 │
│  · RoPE: token loop → head×pair unroll             │
│  · MHA: Q×K^T→softmax→×V per head                  │
└──────────────────────┬──────────────────────────────┘
                       │ 调用
┌──────────────────────▼──────────────────────────────┐
│  Layer 2: Code Generator (有状态，后端无关接口)      │
│                                                     │
│  状态:                                              │
│  · VRegFile: 虚拟寄存器 → 绑定值追踪                │
│  · BufferMap: scratchpad/output/weight 分配追踪     │
│  · LoopStack: 当前循环嵌套上下文                    │
│                                                     │
│  接口:                                              │
│  · bind_ptr(vreg, source) → 加载指针               │
│  · emit_trace(body: &[TraceOp], regs) → 发射计算   │
│  · emit_loop(bound, body_fn) → 发射循环            │
│  · alloc_buffer(size, label) → 分配 scratchpad     │
│  · save_restore(vregs, body_fn) → 保护寄存器       │
│  · verify() → 验证状态一致性                        │
└──────────────────────┬──────────────────────────────┘
                       │ 委托
┌──────────────────────▼──────────────────────────────┐
│  Layer 1: Backend Emitter (后端特化，每 ISV 实现)    │
│                                                     │
│  x86_64:                                            │
│  · VReg→物理寄存器映射 (Activation→r13, Acc→ymm0)  │
│  · TraceOp→SIMD 指令 (Add→vaddps, Exp→多项式)     │
│  · 特化: AVX-512 mask, AMX tile, VNNI dot          │
│  · ABI: System V AMD64, callee-saved r13/r14       │
│                                                     │
│  aarch64:                                           │
│  · VReg→物理寄存器映射 (Activation→x19, Acc→v0)    │
│  · TraceOp→NEON/SVE 指令 (Add→fadd, Exp→多项式)   │
│  · 特化: SVE scalable vector, SME matrix           │
│  · ABI: AAPCS64, callee-saved x19-x28              │
│                                                     │
│  PTX/HIP/Metal:                                     │
│  · VReg→虚拟寄存器 (%rd0, s0, etc.)               │
│  · TraceOp→GPU 指令 (Add→add.f32, Exp→ex2)        │
│  · 特化: Tensor Core, shared memory, warp shuffle  │
│  · ABI: kernel launch parameters                    │
└─────────────────────────────────────────────────────┘
```

## 3. Layer 1: Backend Emitter trait

每个 ISV 必须实现此 trait。这是与硬件直接交互的唯一层。

```rust
/// 后端代码发射器——每个 ISV (x86/arm/gpu) 实现此 trait。
///
/// 职责：将虚拟操作映射到物理指令。
/// 不做任何算法决策——纯粹的指令翻译 + 寄存器分配。
trait BackendEmitter {
    /// 后端能力声明
    fn capabilities(&self) -> BackendCapabilities;

    // ── 指针加载 ──

    /// 将 source 指向的数据地址加载到虚拟寄存器 vreg。
    /// 后端决定用哪个物理寄存器，返回绑定信息。
    fn load_ptr(&mut self, vreg: VReg, source: PtrSource) -> Result<()>;

    // ── TraceOp 计算 ──

    /// 在向量寄存器上执行一组 TraceOp（elementwise body）。
    /// `width` 指定当前 SIMD 宽度（AVX2=8, AVX-512=16, NEON=4）。
    /// 后端将每个 TraceOp 映射到原生 SIMD 指令。
    fn emit_trace_ops(&mut self, ops: &[TraceOp], width: SimdWidth) -> Result<()>;

    // ── 循环控制 ──

    /// 发射循环头（计数器初始化 + 比较 + 跳转）。
    fn emit_loop_header(&mut self, counter: VReg, bound: LoopBound) -> Result<LoopHandle>;
    /// 发射循环尾（递增 + 回跳）。
    fn emit_loop_footer(&mut self, handle: LoopHandle) -> Result<()>;

    // ── 内存操作 ──

    /// 向量加载：从 base+offset 加载 width 个元素到向量寄存器。
    fn vec_load(&mut self, dst: VReg, base: VReg, offset: usize) -> Result<()>;
    /// 向量存储：将向量寄存器写入 base+offset。
    fn vec_store(&mut self, base: VReg, offset: usize, src: VReg) -> Result<()>;
    /// 标量广播：将标量值广播到向量寄存器。
    fn broadcast(&mut self, dst: VReg, src: ScalarSource) -> Result<()>;

    // ── 寄存器管理 ──

    /// 保存一组虚拟寄存器到栈（push 或 spill）。
    fn save_regs(&mut self, regs: &[VReg]) -> Result<RegSaveHandle>;
    /// 恢复之前保存的寄存器。
    fn restore_regs(&mut self, handle: RegSaveHandle) -> Result<()>;

    // ── 后端特化扩展点 ──

    /// 后端特有的高性能原语。
    /// 返回 None 表示该后端不支持此原语，Layer 3 会用通用路径。
    fn specialized(&mut self) -> Option<&mut dyn BackendSpecialized>;

    // ── 输出 ──

    /// 完成代码生成，返回可执行字节。
    fn finalize(self) -> Result<Vec<u8>>;
}
```

## 4. Backend Capabilities（后端能力声明）

```rust
struct BackendCapabilities {
    /// SIMD 宽度选项（后端可能支持多个宽度）
    simd_widths: Vec<SimdWidth>,
    /// 最大可用向量累加器数量（决定 GEMM MR×NR）
    max_accumulators: usize,
    /// 可用临时向量寄存器数量
    max_temp_vectors: usize,
    /// 是否支持 FMA（fused multiply-add）
    has_fma: bool,
    /// 是否支持原生 F16 计算
    has_native_f16: bool,
    /// 是否支持原生 BF16 计算
    has_native_bf16: bool,
    /// 缓存层级信息（L1/L2/L3 大小）
    cache: CacheHierarchy,
    /// 硬件特化能力
    extensions: BackendExtensions,
}

/// 后端特化能力（每个 ISV 不同）
enum BackendExtension {
    // x86
    Avx512Mask,           // k-mask 寄存器
    Amx { tile_m, tile_n, tile_k },  // 矩阵加速
    Vnni,                 // INT8 点积
    Avx512Fp16,          // 原生 FP16
    Bf16Dpbf,            // BF16 dot product

    // aarch64
    Sve { max_vl },      // 可伸缩向量
    Sme { tile_size },   // 矩阵引擎
    DotProd,             // INT8 sdot
    F16Fml,              // FP16 fused multiply-long

    // GPU
    TensorCore { gen, m, n, k },  // Tensor Core 代数
    SharedMemory { size },        // 共享内存
    WarpShuffle,                  // warp 内通信
    AsyncCopy,                    // 异步全局→共享拷贝
}
```

## 5. BackendSpecialized trait（特化扩展点）

```rust
/// 后端特有的高性能路径。
///
/// Layer 3 在通用算法路径中检查后端是否支持特化原语，
/// 支持则调用特化路径，否则用 TraceOp 通用路径。
///
/// 设计原则：特化路径是**优化**，不是**必须**。
/// 移除任何特化路径后，通用路径必须产生正确结果。
trait BackendSpecialized {
    // ── x86 特化 ──

    /// AMX tile GEMM（替代 BLIS FMA 微内核）
    fn amx_tile_gemm(&mut self, m: usize, n: usize, k: usize) -> Result<()> {
        Err("AMX not supported".into())
    }

    /// VNNI INT8 点积（替代 INT8 GEMV 的标量累加）
    fn vnni_dot_product(&mut self, len: usize) -> Result<()> {
        Err("VNNI not supported".into())
    }

    // ── aarch64 特化 ──

    /// SVE predicated 循环（替代固定宽度 NEON 循环 + 标量 tail）
    fn sve_predicated_loop(&mut self, len: usize, body: &[TraceOp]) -> Result<()> {
        Err("SVE not supported".into())
    }

    /// SME outer product（替代 NEON 微内核）
    fn sme_outer_product(&mut self, m: usize, n: usize, k: usize) -> Result<()> {
        Err("SME not supported".into())
    }

    // ── GPU 特化 ──

    /// Tensor Core MMA（替代标量 FMA 循环）
    fn tensor_core_mma(&mut self, m: usize, n: usize, k: usize) -> Result<()> {
        Err("Tensor Core not supported".into())
    }

    /// Shared memory 协作加载（替代全局内存直接读取）
    fn shared_mem_cooperative_load(&mut self, size: usize) -> Result<()> {
        Err("Shared memory not supported".into())
    }
}
```

## 6. Layer 2: Code Generator（有状态，后端无关）

```rust
/// 有状态的代码生成器——控制 TraceOp 到物理指令的映射过程。
///
/// 跨所有后端统一。持有 BackendEmitter 实例 + 状态追踪。
struct CodeGenerator<B: BackendEmitter> {
    backend: B,
    state: KernelState,
}

impl<B: BackendEmitter> CodeGenerator<B> {
    /// 创建新的 per-kernel 生成器
    fn new(backend: B, alloc_bytes: usize) -> Self;

    /// 加载指针到虚拟寄存器（自动追踪状态）
    fn bind_ptr(&mut self, vreg: VReg, source: PtrSource) -> Result<()> {
        self.state.ctx.set(vreg, source.to_binding());
        self.backend.load_ptr(vreg, source)
    }

    /// 发射 TraceOp 计算（自动选择 SIMD 宽度）
    fn emit_trace(&mut self, ops: &[TraceOp]) -> Result<()> {
        let width = self.backend.capabilities().optimal_simd_width();
        self.backend.emit_trace_ops(ops, width)
    }

    /// 发射循环（自动追踪 LoopStack）
    fn emit_loop(&mut self, bound: LoopBound, body: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let counter = self.state.alloc_vreg();
        let handle = self.backend.emit_loop_header(counter, bound)?;
        self.state.push_loop(counter, bound);
        body(self)?;
        self.state.pop_loop();
        self.backend.emit_loop_footer(handle)
    }

    /// 分配 scratchpad 空间（自动追踪水位 + 验证越界）
    fn alloc_buffer(&mut self, size: usize, label: &'static str) -> BufferSlot {
        self.state.scratchpad.alloc(size, label)
    }

    /// 保存/恢复寄存器（自动追踪 context snapshot）
    fn save_restore(&mut self, vregs: &[VReg], body: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let snap = self.state.ctx.snapshot(vregs);
        let handle = self.backend.save_regs(vregs)?;
        body(self)?;
        self.backend.restore_regs(handle)?;
        self.state.ctx.restore(snap);
        Ok(())
    }

    /// 最终验证
    fn finalize(self) -> Result<Vec<u8>> {
        self.state.final_audit();
        self.backend.finalize()
    }
}
```

## 7. Layer 3: Algorithm Driver（后端无关）

```rust
/// 算法驱动器——控制循环结构和融合策略。
/// 完全后端无关，只使用 Layer 2 的 CodeGenerator 接口。
///
/// 这一层决定"做什么"（GEMM 的 tiling、RoPE 的 token 循环、MHA 的 head 迭代）。
/// Layer 1 后端决定"怎么做"（用什么指令实现每一步）。

fn emit_gemm<B: BackendEmitter>(
    gen: &mut CodeGenerator<B>,
    layout: &GemmLayout,
    epilogue: &[TraceOp],
) -> Result<()> {
    let caps = gen.backend.capabilities();
    let (mr, nr, mc, kc, nc) = compute_blocking(layout, &caps);

    // 尝试后端特化路径（AMX / Tensor Core / SME）
    if let Some(spec) = gen.backend.specialized() {
        if let Ok(()) = match () {
            _ if caps.has_extension(BackendExtension::Amx { .. }) =>
                spec.amx_tile_gemm(layout.m, layout.n, layout.k),
            _ if caps.has_extension(BackendExtension::TensorCore { .. }) =>
                spec.tensor_core_mma(layout.m, layout.n, layout.k),
            _ if caps.has_extension(BackendExtension::Sme { .. }) =>
                spec.sme_outer_product(layout.m, layout.n, layout.k),
            _ => Err("no specialization".into()),
        } {
            return Ok(());
        }
    }

    // 通用 BLIS 路径（所有后端共享算法结构）
    gen.emit_loop(LoopBound::Step(0, nc, layout.n), |gen| {          // NC loop
        let pack_b = gen.alloc_buffer(layout.pack_b_bytes(kc, nc, nr), "pack_b");
        gen.emit_pack_b(layout, pack_b)?;
        gen.emit_loop(LoopBound::Step(0, kc, layout.k), |gen| {     // KC loop
            gen.emit_loop(LoopBound::Runtime("seq_len", mc), |gen| { // MC loop (dynamic M)
                let pack_a = gen.alloc_buffer(layout.pack_a_bytes(mc, kc, mr), "pack_a");
                gen.emit_pack_a(layout, pack_a)?;
                // NR loop + microkernel
                gen.emit_loop(LoopBound::Step(0, nr, nc), |gen| {
                    gen.emit_microkernel(mr, nr, kc)?;
                    if !epilogue.is_empty() {
                        gen.emit_trace(epilogue)?;    // epilogue injection
                    }
                    gen.emit_store_accumulators(layout)
                })
            })
        })
    })
}

fn emit_rope<B: BackendEmitter>(
    gen: &mut CodeGenerator<B>,
    layout: &RopeLayout,
    theta: f64,
) -> Result<()> {
    let freqs = precompute_freq_table(layout.half, layout.head_dim, theta);
    gen.emit_loop(LoopBound::Runtime("seq_len", 0), |gen| {     // token loop
        let pos = gen.load_position_for_current_token()?;
        for head in 0..layout.num_heads {
            for pair in 0..layout.half {
                let angle = gen.emit_mul_scalar(pos, freqs[pair]);
                let (cos, sin) = gen.emit_sincos(angle)?;
                let off = layout.pair_byte_offset(head, pair);
                let x0 = gen.vec_load_scalar(VReg::Input, off);
                let x1 = gen.vec_load_scalar(VReg::Input, off + layout.pair_element_stride());
                // out[0] = x0*cos - x1*sin
                // out[1] = x1*cos + x0*sin
                gen.emit_rope_rotation(x0, x1, cos, sin, VReg::Output, off)?;
            }
        }
        Ok(())
    })
}

fn emit_elementwise<B: BackendEmitter>(
    gen: &mut CodeGenerator<B>,
    layout: &ElementwiseLayout,
    body: &[TraceOp],
) -> Result<()> {
    let width = gen.backend.capabilities().optimal_simd_width();
    let vec_count = layout.elem_count / width;
    let tail = layout.elem_count % width;

    // 主循环：SIMD 宽度步进
    gen.emit_loop(LoopBound::Count(vec_count), |gen| {
        gen.vec_load(VReg::Acc(0), VReg::Input, gen.loop_offset())?;
        gen.emit_trace(body)?;
        gen.vec_store(VReg::Output, gen.loop_offset(), VReg::Acc(0))
    })?;

    // 标量 tail
    if tail > 0 {
        gen.emit_scalar_tail(tail, body)
    } else {
        Ok(())
    }
}
```

## 8. 数据流总览

```
scalar_silu()                     ← 算子定义（纯标量 Rust 函数）
    │
    ▼ Phase 0: 符号执行
OpTrace { pattern: Elementwise,
          body: [Input(0), Neg(0), Exp(1), Const(1.0), Add(2,3), Recip(4), Mul(0,5)] }
    │
    ▼ Phase 1-2: 图分析 + 融合决策
FusionPlan { groups: [GEMM(anchor) + Epilogue(silu)] }
    │
    ▼ Phase 3: 代码生成
Layer 3: emit_gemm(gen, layout, epilogue=[silu_body])
    │
    ├─ gen.emit_loop(NC) → Layer 2 追踪 LoopStack
    │   ├─ gen.alloc_buffer(pack_b) → Layer 2 追踪 BufferMap
    │   ├─ gen.emit_pack_b() → Layer 1 发射 pack 指令
    │   └─ gen.emit_loop(KC)
    │       └─ gen.emit_loop(MC, dynamic)
    │           ├─ gen.emit_microkernel(mr, nr, kc)
    │           │   └─ Layer 1: TraceOp::FMA → vfmadd231ps (x86)
    │           │                             → fmla (aarch64)
    │           │                             → fma.rn.f32 (PTX)
    │           └─ gen.emit_trace(silu_body)   ← epilogue injection
    │               └─ Layer 1: TraceOp::Neg → vxorps (x86)
    │                           TraceOp::Exp → 多项式逼近
    │                           TraceOp::Mul → vmulps
    │
    ▼
CodeGenerator::finalize()
    ├─ KernelState::final_audit() → 验证寄存器/buffer 一致性
    └─ BackendEmitter::finalize() → 返回可执行字节码
```

## 9. 迁移路径

### Phase 1: 定义 trait + 适配现有代码（不改行为）
- 定义 `BackendEmitter` / `BackendSpecialized` / `BackendCapabilities` trait
- 现有 `X86CodeGen` 实现 `BackendEmitter`（包装现有 emit 方法）
- `CodeGenerator<X86CodeGen>` 包装现有 `emit_plan`
- 所有测试不变

### Phase 2: 逐算法迁移到 Layer 3 driver
- `emit_elementwise` → 通用 driver（后端无关的 SIMD 循环）
- `emit_rope` → 通用 driver（token loop + pair unroll）
- `emit_norm` → 通用 driver（reduce + finalize + transform）
- 每迁移一个算法，删除后端中对应的硬编码实现

### Phase 3: GEMM 迁移
- BLIS GEMM 的 NC/KC/MC/NR 循环结构提升到 Layer 3
- pack_a/pack_b 内部通过 BackendEmitter 发射
- microkernel 通过 BackendEmitter + TraceOp::FMA 发射
- 特化路径（AMX/Tensor Core/SME）通过 BackendSpecialized

### Phase 4: aarch64 + GPU 后端
- 实现 `AArch64Emitter: BackendEmitter`
- 实现 `PtxEmitter: BackendEmitter`
- Layer 3 driver 自动适配——同一份算法代码，不同后端输出

## 10. 核心约束

1. **Layer 3 零后端代码**: 算法 driver 中禁止出现 `mov`/`vaddps`/`fadd` 等物理指令。只通过 `gen.emit_trace()` / `gen.vec_load()` / `gen.emit_loop()` 操作。

2. **Layer 1 零算法逻辑**: BackendEmitter 中禁止出现循环结构决策、分块策略、融合模式判断。只做 TraceOp → 物理指令映射。

3. **Layer 2 零假设**: CodeGenerator 不假设任何后端细节——不知道有几个寄存器、不知道 SIMD 宽度、不知道栈布局。所有信息通过 `capabilities()` 查询。

4. **TraceOp 是 SSOT**: 所有算子的计算语义由 `Vec<TraceOp>` 唯一定义。任何后端产生的数值结果必须与标量参考实现一致。

5. **特化是优化不是必须**: `BackendSpecialized` 的每个方法默认返回 `Err`。移除任何特化后，通用 TraceOp 路径必须产生正确结果。

6. **状态机不可跳过**: 每条指令的发射必须经过 `CodeGenerator`，由 `KernelState` 追踪。直接操作 `asm` 绕过生成器 = 违规。

## 11. 三代硬件能力矩阵

> 生成器和优化器必须覆盖以下所有硬件代际的指令和机制。

### 11.1 NVIDIA (SM90 → SM100 → SM130)

| 能力 | Hopper SM90 (H100) | Blackwell SM100 (B200) | Blackwell SM120 (GB10) | Rubin SM130 (R100, 2026Q4) |
|------|---------------------|------------------------|------------------------|---------------------------|
| **矩阵指令** | `wgmma.mma_async` (warp group MMA) | `tcgen05.mma` (单线程发射，7 条新指令，2-4x wgmma) | `mma.sync` 扩展版 (无 tcgen05) | 预计继承 tcgen05 + 新一代 |
| **精度** | FP16/BF16/FP8/INT8/TF32 | + FP4/FP6 原生 | + FP4/FP6 原生 | + 预计更多微精度格式 |
| **专用内存** | 无 | **TMEM** (256KB/SM, tensor core 专用) | 无 TMEM | 预计扩展 TMEM |
| **异步机制** | TMA (Tensor Memory Accelerator) | TMA + tcgen05 async pipeline | TMA (无 tcgen05) | TMA 增强 |
| **线程模型** | warp 同步 (32 线程协作) | 单线程 MMA (去 warp 同步) | warp 同步 mma.sync | 预计继承单线程 |
| **互连** | NVLink 4 | NVLink 5 | 无 NVLink | NVLink 6 (NVL576) |
| **HBM** | HBM3 80GB 3.35TB/s | HBM3E 192GB 8TB/s | LPDDR5X | HBM4 288GB |

#### 生成器影响
- `BackendSpecialized::TensorCoreMMA` 必须区分 wgmma (SM90) vs tcgen05 (SM100) vs mma.sync (SM120)
- `TraceOp` 需要 `QuantFMA { act_bits, weight_bits }` 表达 FP4×FP8 混合精度
- TMEM 引入新的内存层级：`MemRegion::TensorMem` (仅 SM100)
- 异步流水线需要 `AsyncPipeline` trait 区分 TMA vs tcgen05 async

### 11.2 AMD (CDNA3 → CDNA4 → CDNA5)

| 能力 | CDNA3 gfx942 (MI300) | CDNA4 gfx950 (MI325) | CDNA5 (MI400, 2026) |
|------|----------------------|----------------------|---------------------|
| **矩阵指令** | `v_mfma_f32_*` (wave64 协作) | `v_mfma_*` + **block exponent scaling** | 预计增强 MFMA |
| **精度** | FP16/BF16/FP8/INT8 | + FP6(E2M3/E3M2)/FP4(E2M1) 原生 | + FP4/FP6/BF16 增强 |
| **Scaling 指令** | 无 | `__builtin_amdgcn_mfma_scale_f32_*` (block-scaled MMA) | 预计增强 block scaling |
| **Wave 模式** | wave64 | wave64 | 预计 wave64 |
| **HBM** | HBM3 192GB | HBM3E | HBM4 432GB 19.6TB/s |
| **FLOPS** | 1.3 PFLOPS FP8 | ~2.6 PFLOPS FP8 | 40 PFLOPS FP4 / 20 PFLOPS FP8 |

#### RDNA4 (消费级 gfx1200)

| 能力 | RDNA3 gfx1100 | RDNA4 gfx1200 |
|------|---------------|---------------|
| **矩阵指令** | WMMA 16×16 (wave32) | WMMA 16×16 增强 + FP8/BF8 |
| **性能** | 基准 | 16-bit 2x, 8-bit/4-bit 4x |
| **寄存器** | 全宽 VGPR | **半宽 VGPR** (布局不兼容 RDNA3) |

#### 生成器影响
- `BackendSpecialized::MatrixFMA` 必须支持 block-scaled 模式 (`mfma_scale`)
- RDNA4 WMMA 的 VGPR 布局变化需要 **Layout Algebra 感知**（半宽映射）
- `MixedPrecision` 需要支持 FP6/FP4/BF8 等微精度
- wave64 vs wave32 影响 tile 大小和协作模式

### 11.3 Intel (Sapphire Rapids → Granite Rapids → Diamond Rapids)

| 能力 | Sapphire Rapids | Granite Rapids (2024) | Diamond Rapids (2026H2) |
|------|-----------------|----------------------|------------------------|
| **AMX** | AMX-BF16, AMX-INT8 | AMX 增强 | **AMX-FP8, AMX-TF32, AMX-TRANSPOSE, AMX-MOVRS, AMX-AVX512** |
| **AVX** | AVX-512 | AVX10.1/256 | **AVX10.2** (全新指令集) |
| **新指令** | VNNI | + AVX-VNNI-INT8 | + **MOVRS** (read-shared hint), SHA512, SM3/SM4, VNNI-INT16 |
| **Tile** | TILECFG + TDPBF16PS | + 增强 | + **TDPFP8PS** (FP8 tile dot product) |
| **核心** | 最多 60 核 | 最多 128 核 | 最多 **256 P-cores** |

#### Panther Lake (2025Q4 客户端)

| 能力 | Arrow Lake | Panther Lake |
|------|------------|-------------|
| **ISA** | AVX10.1 | AVX10.2 (Panther Cove 微架构) |
| **AI** | 基础 NPU | 增强 NPU + AVX10.2 AI 指令 |
| **制程** | Intel 20A | **Intel 18A** |

#### 生成器影响
- `BackendSpecialized::AmxTileGemm` 必须区分 BF16/INT8 (SPR) vs FP8/TF32 (DMR)
- AMX-TRANSPOSE 允许 tile 内转置，影响 pack_b 策略
- AMX-MOVRS 提供 read-shared memory hint，影响预取策略
- AVX10.2 新指令需要扩展 TraceOp→x86 映射表
- AMX-AVX512 桥接指令允许 AVX-512 和 AMX 混用

### 11.4 ARM (Neoverse V2 → V3 → V4)

| 能力 | Neoverse V2 (Grace) | Neoverse V3 / Cortex-X5 | Neoverse V4 / C1 (2025) |
|------|---------------------|--------------------------|-------------------------|
| **向量** | SVE2 (128-bit fixed) | SVE2 (256-bit) | SVE2 |
| **矩阵** | 无 SME | SME2 | **SME2** (full ZA array) |
| **矩阵指令** | N/A | SMOPA/SMOPS (outer product) | SMOPA/SMOPS 增强 |
| **流模式** | N/A | SMSTART/SMSTOP streaming mode | SMSTART/SMSTOP |
| **精度** | FP16/BF16/INT8 | + FP8 (FEAT_FP8) | + FP8 |
| **ZA 存储** | N/A | 二维矩阵寄存器 | 增强 ZA |

#### 生成器影响
- `BackendSpecialized::SmeOuterProduct` 需要 SMSTART/SMSTOP 生命周期管理
- SVE 的可伸缩宽度需要 **predicated loop**（不是固定宽度 unroll）
- SME2 的 ZA array 是独立于 NEON/SVE 的第三寄存器文件
- streaming mode 和 non-streaming mode 之间切换有开销，需要优化器决策

## 12. TraceOp 扩展清单

基于三代硬件调研，`TraceOp` 需要以下扩展：

### 12.1 计算语义扩展

```rust
pub enum TraceOp {
    // ── 现有（保留不变）──
    Input(u32), Const(f64),
    Add(u32, u32), Sub(u32, u32), Mul(u32, u32), Div(u32, u32),
    Fma(u32, u32, u32),
    Neg(u32), Abs(u32), Exp(u32), Sqrt(u32), Rsqrt(u32),
    Tanh(u32), Recip(u32), Log(u32),
    Max(u32, u32), Min(u32, u32),
    ConditionalBranch(u32, u32, u32),

    // ── 量化混合精度（gfx950 mfma_scale / SM100 tcgen05 / AMX-FP8）──
    /// 混合精度 FMA：不同位宽的 activation × weight → 累加器
    QuantFma {
        acc: u32,
        act: u32,
        weight: u32,
        act_dtype: QuantPrecision,
        weight_dtype: QuantPrecision,
    },
    /// Block exponent scaling（CDNA4 mfma_scale 原生支持）
    BlockScale {
        data: u32,
        scale: u32,
        block_size: usize,
    },

    // ── 类型转换（F16C / fcvtl / cvt.f32.f16）──
    Cast { src: u32, from: QuantPrecision, to: QuantPrecision },

    // ── 水平归约（vhaddps / faddp / shfl.sync）──
    HReduce { src: u32, op: ReduceKind },

    // ── 内存层级控制 ──
    /// Prefetch hint（prefetcht0 / prfm / prefetch.global）
    Prefetch { level: CacheLevel },
    /// Non-temporal store（vmovntps / stnp / st.cs）
    NonTemporalStore,

    // ── 位操作（量化解包）──
    /// 位域提取（ubfx / bfe / shift+mask）
    BitExtract { src: u32, offset: u32, width: u32 },
    /// Permute/shuffle（vpshufb / tbl / prmt）
    Permute { src: u32, indices: u32 },

    // ── 比较和掩码 ──
    /// 比较生成掩码（vcmpps→kmask / fcmgt→pred / setp）
    Compare { a: u32, b: u32, op: CmpOp },
    /// 掩码应用（后端自动选择 k-mask / SVE predicate / GPU predicate）
    MaskedOp { op: Box<TraceOp>, mask: u32 },
}

/// 量化精度描述
enum QuantPrecision {
    F32, F16, BF16,
    FP8_E4M3, FP8_E5M2,           // NVIDIA/AMD FP8
    FP6_E2M3, FP6_E3M2,           // AMD CDNA4 FP6
    FP4_E2M1,                      // AMD CDNA4/NVIDIA Blackwell FP4
    INT8, INT4,
    TF32,                           // Intel AMX-TF32
}

enum ReduceKind { Sum, Max, Min, Prod }
enum CacheLevel { L1, L2, L3, NonTemporal }
enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }
```

### 12.2 硬件机制扩展（非 TraceOp，通过 BackendSpecialized）

```rust
/// Tile/Matrix 计算原语
trait TileCompute: BackendSpecialized {
    /// 配置 tile 寄存器
    /// - Intel AMX: TILECFG（指定 tile 行列数和类型）
    /// - ARM SME: SMSTART（进入 streaming SVE mode）
    /// - GPU: 无显式配置
    fn tile_configure(&mut self, config: TileConfig) -> Result<()>;

    /// Tile 矩阵乘法
    /// - Intel AMX: TDPBF16PS / TDPFP8PS (Diamond Rapids)
    /// - AMD CDNA: v_mfma_f32_32x32x64 / v_mfma_scale (gfx950)
    /// - ARM SME: FMOPA / SMOPA
    /// - NVIDIA SM100: tcgen05.mma
    /// - NVIDIA SM120: mma.sync (扩展)
    fn tile_gemm(&mut self, config: TileGemmConfig) -> Result<()>;

    /// Tile 转置（Intel AMX-TRANSPOSE, Diamond Rapids）
    fn tile_transpose(&mut self, tile: TileId) -> Result<()> {
        Err("tile transpose not supported".into())
    }

    /// 释放 tile 资源
    fn tile_release(&mut self) -> Result<()>;
}

struct TileGemmConfig {
    m: usize, n: usize, k: usize,
    precision: MixedPrecision,
    /// Block scaling（CDNA4 gfx950 特有）
    block_scaling: Option<BlockScaleConfig>,
}

struct BlockScaleConfig {
    scale_format: QuantPrecision,
    block_size: usize,
}

/// 异步内存流水线
trait AsyncPipeline: BackendSpecialized {
    /// 异步全局→共享内存拷贝
    /// - NVIDIA: TMA (cp.async.bulk)
    /// - AMD: 异步 global→LDS
    fn async_copy(&mut self, dst: MemRegion, src: MemRegion, size: usize) -> Result<AsyncHandle>;

    /// 异步等待
    fn async_wait(&mut self, handle: AsyncHandle) -> Result<()>;

    /// 流水线阶段屏障
    /// - NVIDIA SM100: arrive.expect_tx + barrier.wait
    fn pipeline_barrier(&mut self, stage: PipelineStage) -> Result<()>;

    /// Warp/Wave 协作同步
    /// - NVIDIA: __syncwarp / bar.sync
    /// - AMD: s_barrier
    fn cooperative_sync(&mut self, scope: SyncScope) -> Result<()>;
}

/// 专用内存层级
trait SpecializedMemory: BackendSpecialized {
    /// NVIDIA SM100 TMEM（Tensor Memory, 256KB/SM）
    fn tmem_store(&mut self, offset: usize, data: VReg) -> Result<()> {
        Err("TMEM not supported".into())
    }
    fn tmem_load(&mut self, dst: VReg, offset: usize) -> Result<()> {
        Err("TMEM not supported".into())
    }

    /// ARM SME ZA Array（二维矩阵寄存器）
    fn za_store(&mut self, tile: u32, row: usize, src: VReg) -> Result<()> {
        Err("ZA not supported".into())
    }
    fn za_load(&mut self, dst: VReg, tile: u32, row: usize) -> Result<()> {
        Err("ZA not supported".into())
    }
}

/// 高级向量操作
trait AdvancedVector: BackendSpecialized {
    /// SVE predicated loop（可伸缩宽度，不需要标量 tail）
    fn predicated_loop(&mut self, len: usize, body: &[TraceOp]) -> Result<()> {
        Err("predicated loop not supported".into())
    }

    /// 整数点积（VNNI / sdot / dp4a）
    fn integer_dot_product(&mut self, a: VReg, b: VReg, acc: VReg, bits: u8) -> Result<()> {
        Err("integer dot product not supported".into())
    }

    /// Read-shared memory hint（Intel MOVRS, Diamond Rapids）
    fn read_shared_hint(&mut self, addr: VReg) -> Result<()> {
        Err("MOVRS not supported".into())
    }
}

enum MemRegion {
    Global,
    Shared,           // GPU shared memory / CPU L1
    TensorMem,        // NVIDIA SM100 TMEM (256KB/SM)
    ZaArray,          // ARM SME ZA matrix storage
    Register,
}

enum SyncScope {
    Warp,             // NVIDIA warp (32 threads)
    Wave,             // AMD wave (64 threads)
    Workgroup,        // GPU workgroup / block
    Device,
}
```

## 13. 优化器影响

硬件代际特性不仅影响 Layer 1 的指令选择，还影响 Phase 2 的融合决策：

### 13.1 融合策略的硬件自适应

| 决策 | SM90 | SM100 | CDNA4 | Diamond Rapids | ARM SME2 |
|------|------|-------|-------|----------------|----------|
| **GEMM 分块** | wgmma tile (64×128×16) | tcgen05 tile (可变) | mfma tile (32×32×K) | AMX tile (16×16) | ZA outer product |
| **Epilogue 融合深度** | 受 warp 同步限制 | 单线程发射→更深融合 | wave64 限制 | AVX10.2 寄存器充裕 | streaming mode 切换开销 |
| **内存预取** | TMA 自动 | TMA + TMEM staging | LDS prefetch | MOVRS read-shared | SVE gather-prefetch |
| **量化 GEMM** | FP8 WGMMA | FP4×FP8 tcgen05 一条指令 | mfma_scale block-scaled | AMX-FP8 tile dot | 无原生量化矩阵 |
| **Pack 策略** | 全局→shared→register | 全局→TMEM→register (新路径) | 全局→LDS→VGPR | RAM→L1→tile regs | RAM→ZA array |

### 13.2 Layer 3 Algorithm Driver 的硬件感知

```rust
fn emit_gemm<B: BackendEmitter>(gen: &mut CodeGenerator<B>, layout: &GemmLayout, ...) {
    let caps = gen.backend.capabilities();

    // 优先级 1: Tile/Matrix 硬件
    if let Some(tile) = gen.backend.tile_compute() {
        let config = TileGemmConfig::from_layout_and_caps(layout, &caps);

        // SM100: tcgen05 + TMEM staging
        if caps.has_tmem() {
            return emit_tcgen05_gemm_with_tmem(gen, tile, layout);
        }
        // SM90: wgmma + TMA
        if caps.has_async_pipeline() {
            return emit_wgmma_gemm_pipelined(gen, tile, layout);
        }
        // AMX: tile GEMM + AVX-512 epilogue
        if caps.has_amx() {
            return emit_amx_tile_gemm(gen, tile, layout);
        }
        // SME: outer product + ZA accumulate
        if caps.has_sme() {
            return emit_sme_outer_product_gemm(gen, tile, layout);
        }
        // CDNA: v_mfma + block scaling
        if let Some(block_scale) = config.block_scaling {
            return emit_mfma_scaled_gemm(gen, tile, layout, block_scale);
        }
        return emit_tile_gemm_generic(gen, tile, config);
    }

    // 优先级 2: SVE predicated（ARM without SME）
    if let Some(adv) = gen.backend.advanced_vector() {
        if caps.has_sve() {
            return emit_sve_gemm(gen, adv, layout);
        }
    }

    // 优先级 3: 通用 BLIS + TraceOp FMA
    emit_blis_gemm(gen, layout, epilogue)
}
```

## 14. SPEC 优化算法 → 生成器映射

> 将 02-ARCHITECTURE.md §9-§16 中规划的全部优化算法映射到生成器架构的三层中。

### 14.1 §9 Mega-Kernel 块级路由 (ARCH-MEGA-KERNEL)

| 优化 | 生成器层级 | 实现机制 |
|------|-----------|---------|
| **单一 Kernel Launch** | Layer 3 | `emit_mega_kernel()` 将全层算法链（RmsNorm→QKV→RoPE→MHA→FFN→Residual）打包为单一 `CodeGenerator::emit_plan()` |
| **Thread Block 独立路由** | Layer 1 GPU | `BackendSpecialized::cooperative_dispatch()` 生成 per-block Request_State_Table 查表指令 |
| **Hot JMP Patching** | Layer 2 | `KernelState::hotpatch_points` 记录可热修补位置；`CodeGenerator::emit_hotpatch_nop_sled()` 在关键决策点生成 5-byte NOP sled |
| **JIT Director Daemon** | Layer 3 | 外部系统，不在 codegen 内。但 codegen 在 Epilogue 中写入 telemetry 供 daemon 消费 |

### 14.2 §10 Chunked Prefill 交织调度 (ARCH-CHUNKED-PREFILL)

| 优化 | 生成器层级 | 实现机制 |
|------|-----------|---------|
| **Prefill Chunk + Decode 交织** | Layer 3 | `emit_mha()` 根据 `LoopBound::Runtime("total_seq")` 动态处理混合序列 |
| **Decode 零等待** | Layer 3 | `emit_gemm()` 的 MC loop 用 `LoopBound::Runtime("seq_len")` 而非编译时常量 |
| **AdaptiveChunkPolicy** | Layer 3 | 根据 `BackendCapabilities::cache.l2_size` 和运行时序列长度计算最优 chunk 大小 |

### 14.3 §11 TurboQuant 2.0 (ARCH-TURBOQUANT)

| 优化 | 生成器层级 | 实现机制 |
|------|-----------|---------|
| **在线 FWHT 旋转** | Layer 3 + TraceOp | 新增 `TraceOp::FWHT { dim }` 表达 Walsh-Hadamard 变换语义；Layer 3 在 3 个白嫖点（Attention Epilogue、SwiGLU Epilogue、KV Write）调用 `gen.emit_trace(&[fwht_body])` |
| **KV 非对称量化** | Layer 3 + TraceOp | `TraceOp::QuantFma` 表达混合精度 K per-channel / V per-token 量化；`gen.emit_kv_write()` 内联量化指令 |
| **RaBitQ 无偏修正** | Layer 3 | `gen.emit_trace(&[fma_correction])` 在 Attention QK^T 计算后追加 1 条 FMA |
| **双轨显存池** | Layer 2 | `KernelState::memory_map` 追踪主池（3-4bit）和 QJL 校验池（1bit）的地址绑定 |
| **Block Scaling** | TraceOp + Layer 1 | `TraceOp::BlockScale` → AMD gfx950 `mfma_scale` / NVIDIA tcgen05 / Intel AMX-FP8 |

### 14.4 §12 空间异构 (ARCH-SPATIAL-DISAGGREGATION)

| 优化 | 生成器层级 | 实现机制 |
|------|-----------|---------|
| **动态块式计算图** | Layer 3 | `emit_plan()` 接收 `FusionPlan` 中的 `SystemTopology` 切割信息，按物理块分发不同子图 |
| **NUMA 感知** | Layer 1 | `BackendCapabilities::topology` 包含 NUMA 节点信息；Layer 3 据此决定内存分配亲和性 |
| **PCIe/RDMA 边界** | Layer 2 | `KernelState::memory_map` 区分本地显存和远程显存地址空间 |

### 14.5 §13 Epilogue 白嫖网络 (ARCH-EPILOGUE-FUSIONS)

**这是生成器架构最核心的消费者——11 个白嫖点全部通过 Layer 3 的 Epilogue injection 实现。**

| 白嫖点 | Layer 3 emit 调用 | TraceOp 扩展 |
|--------|-------------------|-------------|
| §13.1 Gate-First Skip | `gen.emit_gemm(layout, epilogue=[silu_body + dead_neuron_mask])` | `TraceOp::Compare` + `TraceOp::MaskedOp` |
| §13.2 Centroid 预取 | `gen.emit_softmax(layout, epilogue=[centroid_extract])` → `gen.emit_prefetch(centroid_addr, L2)` | `TraceOp::HReduce { op: ArgMax }` + `TraceOp::Prefetch { L2 }` |
| §13.3 残差旁路 | `gen.emit_elementwise(layout, [add_body + delta_rho])` → 条件 `gen.emit_jump(END_OF_LAYER)` | `TraceOp::Compare` 生成 $\Delta\rho < \varepsilon$ 判断 |
| §13.4 FWHT 旋转 | `gen.emit_kv_write(layout, epilogue=[fwht_body])` | `TraceOp::FWHT { dim }` |
| §13.5 死神经元掩码 | GEMM epilogue 内追加 `[Compare(sigmoid, eps), HReduce(mask, Count)]` | `TraceOp::Compare` + `TraceOp::HReduce` |
| §13.6 MoE 命中计数 | TopK epilogue 内追加 `[AtomicAdd(expert_counter, 1)]` | 新增 `TraceOp::AtomicAdd` |
| §13.7 行级激活统计 | GEMM epilogue 内追加 `[HReduce(row, Sum), HReduce(row, Max)]` | `TraceOp::HReduce` |
| §13.8 RmsNorm Scale | Norm reduce 阶段追加 `[Max(channel_abs)]` | 复用 `TraceOp::Max` |
| §13.9 Softmax 锐度 | Softmax epilogue 内追加 `[Div(max, sum)]` | 复用 `TraceOp::Div` |
| §13.10 Embedding 范数 | Embedding copy 阶段追加 `[HReduce(sq, Sum), Sqrt]` | `TraceOp::HReduce` + `TraceOp::Sqrt` |
| §13.11 残差余弦 | Residual epilogue 追加 `[Fma(dot), Div, Compare]` | 复用现有 TraceOp |

### 14.6 §14 旧世代优化突变 (ARCH-LEGACY-METAMORPHOSIS)

| 优化 | 生成器层级 | 实现机制 |
|------|-----------|---------|
| **Static KV Cache → Paged** | Layer 2 | `KernelState::memory_map` 追踪页表映射 |
| **Token-level → Block-level** | Layer 3 | `emit_mha()` 的 K/V 访问通过 `gen.paged_load()` 替代连续 load |
| **Greedy Sampling → Speculative** | Layer 3 外部 | 不在 codegen 内 |

### 14.7 §15 MoE 异构专家 (ARCH-MOE-EXTREME)

| 优化 | 生成器层级 | 实现机制 |
|------|-----------|---------|
| **冷板凳 Deopt** | Layer 3 + Hot Patch | §13.6 命中计数 → JIT Director → `hotpatch_nop_sled` 封杀冷专家路径 |
| **异步预取掩蔽** | Layer 1 GPU | `AsyncPipeline::async_copy()` 预取下一专家权重 |
| **容错降级** | Layer 3 | 专家执行失败时 `gen.emit_expert_fallback()` 用次优专家替代 |

### 14.8 §16 残差总线四大应用 (ARCH-RESIDUAL-BUS-APPLICATIONS)

| 应用 | 生成器层级 | 实现机制 |
|------|-----------|---------|
| **KV 侧载注入** | Layer 2 | `KernelState::memory_map` 将外部 KV 页表映射到注意力计算路径 |
| **晚期特征融合** | Layer 3 | `gen.emit_residual_add()` 在指定层注入外部特征向量 |
| **LoRA 运行时挂载** | Layer 3 | `gen.emit_gemm()` 的 epilogue 追加 LoRA delta 矩阵乘 |
| **安全探针 (GuardProbe)** | Layer 3 | `gen.emit_guard_probe()` 在目标层 Epilogue 插入线性分类器 |

### 14.9 新增 TraceOp 完整清单（含 §9-§16 需求）

上述优化需要的 TraceOp 扩展汇总：

```rust
// §11 TurboQuant
TraceOp::FWHT { dim: usize },              // Fast Walsh-Hadamard Transform
TraceOp::BlockScale { data, scale, block_size }, // Block exponent scaling

// §13 Epilogue 白嫖网络
TraceOp::AtomicAdd { addr: u32, val: u32 },      // 原子加（MoE 命中计数）
TraceOp::Prefetch { level: CacheLevel },          // 预取（质心引导）
TraceOp::NonTemporalStore,                         // 绕过缓存写（KV 大量写入）

// §12 量化混合精度（跨 §11 + §13）
TraceOp::QuantFma { acc, act, weight, act_dtype, weight_dtype },
TraceOp::Cast { src, from, to },

// §14 通用增强
TraceOp::HReduce { src, op: ReduceKind },         // 水平归约（sum/max/count）
TraceOp::Compare { a, b, op: CmpOp },             // 比较生成掩码
TraceOp::MaskedOp { op: Box<TraceOp>, mask },     // 掩码操作
TraceOp::BitExtract { src, offset, width },        // 位域提取（量化解包）
TraceOp::Permute { src, indices },                 // 排列/洗牌
```

### 14.10 优化覆盖验证矩阵

| 优化 (SPEC §) | TraceOp 覆盖 | Layer 3 Driver | Layer 1 特化 | 状态 |
|---------------|-------------|----------------|-------------|------|
| §9 Mega-Kernel | ✅ hotpatch NOP sled | ✅ emit_mega_kernel | ✅ cooperative_dispatch | 设计完成 |
| §10 Chunked Prefill | ✅ Runtime LoopBound | ✅ emit_mha dynamic seq | N/A | 设计完成 |
| §11.1 FWHT | 🆕 TraceOp::FWHT | ✅ 3 个白嫖点内联 | 各后端多项式/查表 | 设计完成 |
| §11.2 KV 非对称量化 | ✅ QuantFma + Cast | ✅ emit_kv_write | ✅ TileCompute 混合精度 | 设计完成 |
| §11.3 RaBitQ | ✅ FMA | ✅ Attention epilogue | N/A | 设计完成 |
| §11.5 双轨池 | N/A (内存管理) | ✅ memory_map | N/A | 设计完成 |
| §12 空间异构 | N/A | ✅ topology-aware emit | ✅ NUMA/RDMA | 设计完成 |
| §13.1 Gate-First Skip | ✅ Compare + MaskedOp | ✅ GEMM epilogue | N/A | 设计完成 |
| §13.2 Centroid 预取 | 🆕 Prefetch | ✅ Softmax epilogue | ✅ prefetch 指令 | 设计完成 |
| §13.3 残差旁路 | ✅ Compare | ✅ Residual epilogue | N/A | 设计完成 |
| §13.4 FWHT 旋转 | 🆕 FWHT | ✅ KV Write epilogue | N/A | 设计完成 |
| §13.5 死神经元 | ✅ Compare + HReduce | ✅ SiLU epilogue | N/A | 设计完成 |
| §13.6 MoE 命中 | 🆕 AtomicAdd | ✅ TopK epilogue | ✅ 原子指令 | 设计完成 |
| §13.7 行级统计 | ✅ HReduce | ✅ GEMM epilogue | N/A | 设计完成 |
| §13.8 RmsNorm Scale | ✅ Max | ✅ Norm epilogue | N/A | 设计完成 |
| §13.9 Softmax 锐度 | ✅ Div | ✅ Softmax epilogue | N/A | 设计完成 |
| §13.10 Embedding 范数 | ✅ HReduce + Sqrt | ✅ Embedding epilogue | N/A | 设计完成 |
| §13.11 残差余弦 | ✅ FMA + Div | ✅ Residual epilogue | N/A | 设计完成 |
| §13.12 硬件拓扑 | N/A | ✅ topology-aware fusion | ✅ 12 Profile | 设计完成 |
| §14 旧代突变 | N/A | ✅ Paged attention | N/A | 设计完成 |
| §15 MoE 极致 | ✅ AtomicAdd | ✅ expert dispatch | ✅ async_copy | 设计完成 |
| §16 残差总线 | ✅ FMA (LoRA delta) | ✅ residual injection | N/A | 设计完成 |
