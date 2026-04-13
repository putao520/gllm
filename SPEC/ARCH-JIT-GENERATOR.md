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
