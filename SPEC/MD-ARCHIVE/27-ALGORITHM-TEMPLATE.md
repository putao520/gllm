# 27 — 算法模板声明式表达 (Algorithm Template Declarative Expression)

> **铁律 ARCH-ALGO-TEMPLATE**: 算法实现 = 纯数据模板 + TraceOp 归一管线。禁止手写 emit 函数。禁止绕过 auto_select 发射 VmInstr。

## §0 问题与设计哲学

### §0.1 问题根因

`plan_lower.rs` 中 12,891 行代码包含 15+ 手写 `emit_*_inline` 函数，每个函数都是 Rust 命令式代码：

```rust
// 当前: 手写命令式代码生成
fn emit_gemm_blis_inline(prog, m, n, k, ...) {
    let pack_b = prog.alloc_vreg(...);
    prog.emit(LoopBegin { ... });
    for i in 0..mc/mr {
        prog.emit(LoopBegin { ... });
        // ... 200 行手写 VmInstr 发射
    }
    prog.emit(LoopEnd);
}
```

**问题**：
1. **算法逻辑与代码生成耦合** — BLIS 的 5 层循环嵌套结构硬编码在 Rust 函数中
2. **绕过 auto_select 管线** — 手写 emit 函数直接发射 VmInstr，不走 TraceOp → auto_select 统一路径
3. **两条割裂路径** — 简单算子走 SymExec → TraceOp → auto_select，结构算子绕过管线直接写 VmInstr
4. **无跨算法复用** — naive GEMM、BLIS GEMM、GPU tiled GEMM 各自独立函数，重复 80% 逻辑
5. **无法外部编辑** — 添加新算法策略 = 写新 Rust 函数 + 编译，不能数据驱动扩展

### §0.2 设计目标：归一管线

**核心原则：TraceOp 是唯一 IR，auto_select 是唯一 TraceOp→VmInstr 桥梁。所有算法，无论来源，都经过同一条管线。**

```
                    ┌─ SymExec (自动提取) ──────────┐
                    │   Scalar fn → TraceOp ────────┤
TraceOp 来源 ───────┤                               ├→ auto_select → VmInstr → ISA → 机器码
                    └─ 模板解释器 (声明式) ──────────┤
                        AlgoTemplate → TraceOp ─────┘
                                              ↑
                                         统一 IR
                                    (唯一 TraceOp→VmInstr 桥)
```

**JIT 从 auto_select 往下是纯执行器**——不知道自己在编译 GEMM 还是 FFT，只看到 TraceOp 流进来，VmInstr 流出去。

| 来源 | 怎么产 TraceOp | 适用场景 |
|------|-------------|---------|
| SymExec | 自动 trace scalar fn | 逐元素/归约/量化 |
| 模板解释器 | 按模板数据组装 | GEMM/Attention/MoE/RoPE/Norm |
| 手动构造 | 测试中直接写 | 单元测试 |

**三种来源，一种 IR，一条管线。**

### §0.3 与 SPEC 26 META 模式的关系

SPEC 26 建立了 META enum 模式用于参数化单一指令：

```rust
// SPEC 26: 指令级参数化
QuantBlockLoad { mode: BlockUnpackMode }  // 7 种解包模式
DotProduct { dtype: DotDtype }            // 5 种点积类型
```

SPEC 27 将 META 模式扩展到 **算法级**：

```
SPEC 26:  VmInstr + META enum  →  参数化单一指令
SPEC 27:  AlgoTemplate + META  →  参数化整个算法结构，输出到统一 TraceOp IR
```

## §1 核心数据模型

### §1.1 AlgoTemplate

```rust
/// 算法模板 — 纯静态数据，描述算法的循环嵌套结构和算法步骤。
///
/// 模板解释器遍历步骤树，输出 Vec<TraceOp>，
/// 然后走统一 auto_lower_trace → VmInstr 管线。
///
/// 文件位置: algo_templates/*.rs，只包含 static 数据定义，零代码生成逻辑。
pub struct AlgoTemplate {
    /// 模板唯一标识
    pub name: &'static str,
    /// 算法族 META — 标识算法类型和策略变体
    pub strategy: AlgoStrategy,
    /// 设备需求 — 此模板适用的最小设备能力
    pub device_req: DeviceReq,
    /// 算法步骤树 — 描述循环嵌套和计算步骤
    pub steps: &'static [AlgoStep],
    /// 参数表 — tile 大小、unroll 因子等
    pub params: &'static [(&'static str, AlgoParam)],
    /// 依赖的微核定义 (GEMM 类模板)
    pub micro_kernel: Option<&'static MicroKernelDef>,
}
```

### §1.2 AlgoStrategy — 算法族 META

```rust
/// 算法族 META — 类似 BlockUnpackMode，描述算法级策略变体。
///
/// 每个变体对应一个或多个 AlgoTemplate 实例（不同设备特化）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgoStrategy {
    // ── GEMM ──
    GemmNaive,          // Naive 3-loop: ijk, no tiling
    GemmBlis,           // BLIS 5-loop: mc/mr + nc/nr + kc + pack_b + micro-kernel
    GemmGpuTiled,       // GPU smem tiled: load tile → MMA → store
    GemmGpuPipelined,   // GPU async pipeline: cp.async + double buffering
    GemmHardwareTile,   // AMX/SME2: hardware tile register (16×16)

    // ── Attention ──
    AttnMha,            // 标准 MHA: Q×K^T softmax → ×V
    AttnGqa,            // GQA: 共享 K/V heads
    AttnMla,            // MLA: 低秩吸收 (DeepSeek)

    // ── MoE ──
    MoeRouterTopk,      // Router GEMV + Top-K dispatch
    MoePackedDispatch,  // Packed expert dispatch

    // ── Norm ──
    NormRms,            // RmsNorm: 3-phase (reduce → finalize → transform)
    NormLayer,          // LayerNorm: mean + var + normalize

    // ── RoPE ──
    RopeStandard,       // 标准 RoPE: cos/sin rotation
    RopePartial,        // Partial RoPE (p-RoPE): partial dimension rotation
}
```

### §1.3 AlgoStep — 算法步骤树

```rust
/// 算法步骤 — 模板的主体，描述循环嵌套和计算步骤。
///
/// 解释器遍历步骤树，将每个步骤翻译为结构化 TraceOp 序列。
/// 计算类步骤 (TraceBody/FmaReduce/Softmax) 直接产 TraceOp，
/// 结构类步骤 (Loop/LoadPanel/MicroKernel) 组织 TraceOp 的嵌套和上下文。
#[derive(Debug, Clone)]
pub enum AlgoStep {
    // ── 控制结构 ──

    /// 顺序执行一组步骤
    Seq(&'static [AlgoStep]),

    /// 循环: bound/step 从参数表解析
    /// 解释器产 TraceOp::Loop { bound, step, body }
    Loop {
        bound: &'static str,
        step: &'static str,
        body: &'static [AlgoStep],
    },

    /// 条件执行: 仅当设备满足 requirement 时执行 body
    Conditional {
        requirement: DeviceReq,
        body: &'static [AlgoStep],
    },

    // ── 内存操作 (映射到现有 TraceOp) ──

    /// 加载矩阵面板 → TraceOp::PanelLoad
    LoadPanel {
        matrix: MatrixRole,
        rows_param: &'static str,
        cols_param: &'static str,
    },

    /// Pack buffer → TraceOp::PackBuffer
    PackBuffer {
        buffer_name: &'static str,
        rows_param: &'static str,
        cols_param: &'static str,
    },

    /// 微核计算: mr × nr × kc MAC 循环
    /// 解释器展开 MicroKernelDef 为 TraceOp 序列
    MicroKernel,

    /// 存储结果 → TraceOp::PanelStore
    StoreResult {
        rows_param: &'static str,
        cols_param: &'static str,
    },

    // ── GPU 专用 (映射到 TraceOp 扩展) ──

    /// TraceOp::SharedMemDeclare
    SharedMemDeclare { name: &'static str, size_param: &'static str },

    /// TraceOp::AsyncCopyToShared
    AsyncCopyToSmem { buffer_name: &'static str, size_param: &'static str },

    /// TraceOp::AsyncWaitGroup
    AsyncWait { group: u32 },

    /// TraceOp::SyncBarrier
    Barrier { barrier_name: &'static str },

    // ── Tile/Matrix 专用 ──

    TileConfig { rows: &'static str, cols: &'static str },
    TileMma,
    TileRelease,

    // ── 计算步骤 (直接产 TraceOp) ──

    /// 嵌入已有 TraceOp 序列 (来自 SymExec trace 或手动定义)
    TraceBody(&'static [AlgoTraceStep]),

    /// 归约: TraceOp::HReduce
    Reduce { op: ReduceOp },

    /// 激活函数: TraceOp 查表 (Silu→Sigmoid+Mul, Gelu→Exp+...)
    Activation { kind: ActivationKind },

    /// Softmax 三阶段: reduce_max → exp_sum → normalize
    Softmax,

    /// 量化反量化: TraceOp + BlockUnpackMode
    Dequantize { mode: BlockUnpackMode },

    /// 嵌入查找: TraceOp::ScalarLoad + StrideMul + PtrAdd + VecLoadIndexed
    EmbeddingGather,

    /// MoE Router
    MoeRouterGemv { num_experts: &'static str, hidden: &'static str },
    MoeTopK { num_experts: &'static str, top_k: &'static str },

    /// Epilogue 注入
    Epilogue { ops: &'static [EpilogueOp] },
}
```

### §1.4 AlgoParam — 参数来源

```rust
#[derive(Debug, Clone, Copy)]
pub enum AlgoParam {
    Const(usize),
    FromPressureModel(&'static str),
    FromDeviceProfile(&'static str),
    FromGraph(&'static str),
    Derived { base: &'static str, op: ParamArith, operand: usize },
}

#[derive(Debug, Clone, Copy)]
pub enum ParamArith { CeilDiv, Mul, Div, Max, Min }
```

### §1.5 辅助类型

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatrixRole { A, B, C }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceReq {
    CpuAny,       // 任意 CPU
    CpuAvx2,      // AVX2+
    CpuAvx512,    // AVX-512+
    CpuAmx,       // AMX tile
    CpuSme2,      // SME2 tile
    GpuSm70,      // Volta
    GpuSm80,      // Ampere: cp.async
    GpuSm90,      // Hopper: TMA/WGMMA
    GpuSm100,     // Blackwell: FP4
}

impl DeviceReq {
    pub fn priority(&self) -> u32 {
        match self {
            Self::CpuAny => 0, Self::CpuAvx2 => 10, Self::CpuAvx512 => 20,
            Self::CpuAmx => 30, Self::CpuSme2 => 30,
            Self::GpuSm70 => 40, Self::GpuSm80 => 50, Self::GpuSm90 => 60, Self::GpuSm100 => 70,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EpilogueOp {
    BiasAdd, ResidualAdd, Relu, Silu, Gelu,
    RmsNorm { eps_param: &'static str },
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationKind { Relu, Silu, Gelu, Tanh, Sigmoid }
```

### §1.6 MicroKernelDef

```rust
/// 微核定义 — mr × nr × kc 的内层循环。
/// 解释器展开为 TraceOp 序列 (LoadARow/LoadBCol/Fma 重复 mr×nr×k_step 次)。
pub struct MicroKernelDef {
    pub mr: &'static str,
    pub nr: &'static str,
    pub k_step: &'static str,
    pub steps: &'static [MicroKernelStep],
}

pub enum MicroKernelStep {
    LoadARow,        // TraceOp::VecLoadIndexed (A 面板行)
    LoadBCol,        // TraceOp::VecLoadIndexed (B 面板列)
    Fma,             // TraceOp::Fma
    StoreAccumulator, // TraceOp::VecStoreIndexed
    WarpMma,         // TraceOp::WarpMma (GPU 专用)
}
```

## §2 TraceOp 结构型扩展

### §2.1 新增 TraceOp 变体

模板解释器产出的 TraceOp 需要扩展以表达结构型操作。这些扩展也服务于 SymExec 路径（结构型算子的 trace 提取）。

```rust
// 在 trace.rs 的 TraceOp enum 中新增:

// ── 循环结构 (模板解释器和 SymExec 共用) ──

/// 循环: 迭代 bound 次, body 内使用 slot 引用当前迭代。
/// auto_select 处理: 产 LoopBegin/LoopEnd VmInstr。
Loop {
    bound: BoundExpr,
    step_bytes: usize,
    body: Vec<TraceOp>,
},

// ── 结构型内存操作 (GEMM/Attention 面板加载) ──

/// 矩阵面板加载: 从 base + offset 加载 rows×cols 元素到连续寄存器。
/// auto_select 映射: 多条 VecLoad / VecLoadIndexed。
PanelLoad {
    base: u32,        // slot: 基地址
    offset: u32,      // slot: 偏移
    matrix: MatrixRole,
    rows: usize,
    cols: usize,
},

/// 矩阵面板存储: 将连续寄存器写入 base + offset。
/// auto_select 映射: 多条 VecStore。
PanelStore {
    base: u32,
    offset: u32,
    matrix: MatrixRole,
    rows: usize,
    cols: usize,
},

/// Buffer pack: 将 B 面板重排到连续 scratch buffer。
/// auto_select 映射: VecLoad + VecStore 序列。
PackBuffer {
    src: u32,         // slot: 源地址
    dst: u32,         // slot: 目标地址
    rows: usize,
    cols: usize,
    /// 行优先还是列优先 pack
    layout: PackLayout,
},

// ── GPU 结构型 ──

/// 共享内存声明
SharedMemDeclare {
    name: String,
    bytes: usize,
},

/// 异步拷贝到共享内存
AsyncCopyToShared {
    name: String,
    src_offset: u32,
    bytes: usize,
},

/// 等待异步操作组
AsyncWaitGroup { n: u32 },

/// 同步屏障
SyncBarrier { name: String },

// ── Tile 操作 ──

/// 配置硬件 tile 寄存器
TileConfig { rows: usize, cols: usize, dtype: DType },

/// Tile MMA: c += a × b
TileMma { c: u32, a: u32, b: u32 },

/// 释放 tile 资源
TileRelease,

// ── Softmax 三阶段 ──

/// Softmax: reduce_max → exp(x-max) → sum → normalize
/// 展开为已有 TraceOp 组合: HReduce(Max) → Sub → Exp → HReduce(Sum) → Div
Softmax { src: u32, dst: u32 },
```

### §2.2 auto_select 新增映射

每个新增 TraceOp 变体在 `auto_select.rs` 的 `dispatch_trace_op` 中新增一个 match arm：

| TraceOp 变体 | auto_select 映射 | VmInstr 输出 |
|---|---|---|
| `Loop` | 循环展开器 | `LoopBegin` + body + `LoopEnd` |
| `PanelLoad` | 面板加载器 | 多条 `VecLoad` |
| `PanelStore` | 面板存储器 | 多条 `VecStore` |
| `PackBuffer` | pack 发射器 | `VecLoad` + `VecStore` 序列 |
| `SharedMemDeclare` | GPU smem | `SharedMemAlloc` |
| `AsyncCopyToShared` | GPU async | `SharedMemAsyncStore` |
| `AsyncWaitGroup` | GPU wait | `SharedMemAsyncWaitGroup` |
| `SyncBarrier` | GPU barrier | `WarpBarrierArrive` + `WarpBarrierWait` |
| `TileConfig` | tile 配置 | `TileConfig` |
| `TileMma` | tile 计算 | `TileMma` |
| `TileRelease` | tile 释放 | `TileRelease` |
| `Softmax` | softmax 展开 | `HReduce(Max)` + `VecBinOp(Sub)` + `Transcendental(Exp)` + `HReduce(Sum)` + `VecBinOp(Div)` |

**所有映射走同一个 auto_select 函数，不绕过。**

## §3 模板解释器

### §3.1 解释器契约：产出 TraceOp

```rust
/// 模板解释器 — 从纯数据模板生成 TraceOp 序列。
///
/// 解释器是 TraceOp 的生产者之一 (和 SymExec 平级)。
/// 产出的 TraceOp 走统一 auto_lower_trace 管线。
///
/// 解释器不关心:
/// - 最终用什么 ISA 指令 (由 auto_select + ISA lowering 处理)
/// - 具体算法语义 (由模板数据描述)
pub fn instantiate_template(
    template: &AlgoTemplate,
    ctx: &AlgoContext,
) -> Result<Vec<TraceOp>, CompilerError> {
    let resolved = resolve_params(template.params, ctx)?;
    let mut builder = TraceBuilder::new();
    for step in template.steps {
        emit_step(step, &resolved, ctx, &mut builder, template.micro_kernel)?;
    }
    Ok(builder.finish())
}
```

### §3.2 步骤 → TraceOp 翻译

```rust
fn emit_step(
    step: &AlgoStep,
    params: &ResolvedParams,
    ctx: &AlgoContext,
    builder: &mut TraceBuilder,
    mk: Option<&MicroKernelDef>,
) -> Result<(), CompilerError> {
    match step {
        AlgoStep::Seq(steps) => {
            for s in steps { emit_step(s, params, ctx, builder, mk)?; }
        }

        AlgoStep::Loop { bound, step, body } => {
            let bound_val = params.resolve_bound(bound, ctx)?;
            let step_val = params.resolve(step)?;
            let mut body_ops = Vec::new();
            let mut body_builder = TraceBuilder::new();
            for s in body {
                emit_step(s, params, ctx, &mut body_builder, mk)?;
            }
            builder.push(TraceOp::Loop {
                bound: bound_val,
                step_bytes: step_val,
                body: body_builder.finish(),
            });
        }

        AlgoStep::LoadPanel { matrix, rows_param, cols_param } => {
            let base_slot = builder.current_panel_base(*matrix);
            let offset_slot = builder.current_panel_offset(*matrix);
            builder.push(TraceOp::PanelLoad {
                base: base_slot, offset: offset_slot,
                matrix: *matrix,
                rows: params.resolve(rows_param)?,
                cols: params.resolve(cols_param)?,
            });
        }

        AlgoStep::PackBuffer { buffer_name, rows_param, cols_param } => {
            let src = builder.current_b_panel_slot();
            let dst = builder.alloc_scratch_slot(buffer_name);
            builder.push(TraceOp::PackBuffer {
                src, dst,
                rows: params.resolve(rows_param)?,
                cols: params.resolve(cols_param)?,
                layout: PackLayout::ColMajor, // BLIS B 面板列优先
            });
        }

        AlgoStep::MicroKernel => {
            let mk_def = mk.ok_or("MicroKernel without definition")?;
            emit_micro_kernel_as_trace(mk_def, params, builder)?;
        }

        AlgoStep::TraceBody(trace_steps) => {
            // 已有 TraceOp 序列直接注入
            for ts in trace_steps {
                builder.push(ts.to_trace_op(builder.slot_map())?);
            }
        }

        AlgoStep::Softmax => {
            let src = builder.current_accum_slot();
            let dst = builder.alloc_slot();
            builder.push(TraceOp::Softmax { src, dst });
        }

        AlgoStep::Reduce { op } => {
            let src = builder.current_accum_slot();
            let dst = builder.alloc_slot();
            builder.push(TraceOp::HReduce { src, op: match op {
                ReduceOp::Sum => ReduceKind::Sum,
                ReduceOp::Max => ReduceKind::Max,
            }});
        }

        AlgoStep::Barrier { barrier_name } => {
            builder.push(TraceOp::SyncBarrier { name: barrier_name.to_string() });
        }

        AlgoStep::AsyncCopyToSmem { buffer_name, size_param } => {
            let bytes = params.resolve(size_param)?;
            builder.push(TraceOp::AsyncCopyToShared {
                name: buffer_name.to_string(),
                src_offset: builder.current_panel_offset(MatrixRole::A),
                bytes,
            });
        }

        // ... 其他步骤类似翻译为 TraceOp
        _ => { /* 其余 TraceOp 映射 */ }
    }
    Ok(())
}
```

### §3.3 微核 → TraceOp 展开

```rust
/// 微核展开: 将 MicroKernelDef 翻译为 TraceOp 序列。
/// mr 行 × nr 列 × k_step 的 FMA 循环。
fn emit_micro_kernel_as_trace(
    mk: &MicroKernelDef,
    params: &ResolvedParams,
    builder: &mut TraceBuilder,
) -> Result<(), CompilerError> {
    let mr = params.resolve(mk.mr)?;
    let nr = params.resolve(mk.nr)?;
    let k_step = params.resolve(mk.k_step)?;

    let mut body = Vec::new();
    for step in mk.steps {
        match step {
            MicroKernelStep::LoadARow => {
                body.push(TraceOp::VecLoadIndexed { base: builder.a_slot(), offset: builder.k_offset_slot() });
            }
            MicroKernelStep::LoadBCol => {
                body.push(TraceOp::VecLoadIndexed { base: builder.b_slot(), offset: builder.k_offset_slot() });
            }
            MicroKernelStep::Fma => {
                let acc = builder.current_accum_slot();
                let a = builder.last_load_a_slot();
                let b = builder.last_load_b_slot();
                body.push(TraceOp::Fma(acc, a, b));
            }
            MicroKernelStep::StoreAccumulator => {
                body.push(TraceOp::VecLoadIndexed { base: builder.c_slot(), offset: builder.out_offset_slot() });
            }
            MicroKernelStep::WarpMma => {
                // GPU: 直接产 TraceOp::TileMma
                body.push(TraceOp::TileMma { c: builder.acc_slot(), a: builder.a_slot(), b: builder.b_slot() });
            }
        }
    }

    // 微核本身也是 k 维度循环
    builder.push(TraceOp::Loop {
        bound: BoundExpr::Const(k_step),
        step_bytes: params.resolve("elem_bytes")?,
        body,
    });

    Ok(())
}
```

### §3.4 归一调用入口

```rust
/// plan_lower.rs 中的调用方式:
/// 所有算法走同一条 auto_lower_trace 管线。
fn emit_algo_from_template(
    prog: &mut VmProgram,
    strategy: AlgoStrategy,
    ctx: &LoweringContext,
    inputs: &[VRegId],
    width: SimdWidth,
) -> Result<(), CompilerError> {
    // 1. 选择模板
    let template = select_template(strategy, ctx)?;

    // 2. 解释器产出 TraceOp
    let trace_ops = instantiate_template(template, &ctx.into())?;

    // 3. 走统一 auto_select 管线 (和 SymExec trace 完全相同的路径)
    auto_lower_trace(prog, &trace_ops, inputs, width)?;

    Ok(())
}
```

## §4 模板库

### §4.1 文件结构

```
algo_templates/
├── mod.rs          — pub mod 声明 + 注册表 + select_template()
├── gemm.rs         — GEMM 族模板 (5 个)
├── attention.rs    — Attention 族模板 (3 个)
├── norm.rs         — Norm 族模板 (2 个)
├── rope.rs         — RoPE 族模板 (2 个)
├── moe.rs          — MoE 族模板 (2 个)
└── sampling.rs     — Sampling 族模板
```

每个文件只包含 `pub static XXX: AlgoTemplate = AlgoTemplate { ... }` 纯数据定义。

### §4.2 GEMM 模板示例

```rust
// algo_templates/gemm.rs

pub static GEMM_BLIS: AlgoTemplate = AlgoTemplate {
    name: "GemmBlis",
    strategy: AlgoStrategy::GemmBlis,
    device_req: DeviceReq::CpuAvx2,
    steps: &[
        AlgoStep::Loop { bound: "m", step: "mc", body: &[
            AlgoStep::Loop { bound: "n", step: "nc", body: &[
                AlgoStep::Loop { bound: "k", step: "kc", body: &[
                    AlgoStep::PackBuffer { buffer_name: "pack_b", rows_param: "kc", cols_param: "nc" },
                    AlgoStep::Loop { bound: "mc", step: "mr", body: &[
                        AlgoStep::MicroKernel,
                    ]},
                ]},
                AlgoStep::StoreResult { rows_param: "mc", cols_param: "nc" },
            ]},
        ]},
    ],
    params: &[
        ("m", AlgoParam::FromGraph("m")),
        ("n", AlgoParam::FromGraph("n")),
        ("k", AlgoParam::FromGraph("k")),
        ("mc", AlgoParam::FromPressureModel("mc")),
        ("nc", AlgoParam::FromPressureModel("nc")),
        ("kc", AlgoParam::FromPressureModel("kc")),
        ("mr", AlgoParam::FromDeviceProfile("gemm_mr")),
        ("nr", AlgoParam::FromDeviceProfile("gemm_nr")),
    ],
    micro_kernel: Some(&MICRO_KERNEL_BLIS),
};

pub static GEMM_NAIVE: AlgoTemplate = AlgoTemplate {
    name: "GemmNaive",
    strategy: AlgoStrategy::GemmNaive,
    device_req: DeviceReq::CpuAny,
    steps: &[
        AlgoStep::Loop { bound: "m", step: "mr", body: &[
            AlgoStep::Loop { bound: "n", step: "nr", body: &[
                AlgoStep::TraceBody(&[/* zero acc */]),
                AlgoStep::Loop { bound: "k", step: "k_step", body: &[
                    AlgoStep::LoadPanel { matrix: MatrixRole::A, rows_param: "mr", cols_param: "k_step" },
                    AlgoStep::LoadPanel { matrix: MatrixRole::B, rows_param: "k_step", cols_param: "nr" },
                    AlgoStep::MicroKernel,
                ]},
                AlgoStep::StoreResult { rows_param: "mr", cols_param: "nr" },
            ]},
        ]},
    ],
    params: &[
        ("m", AlgoParam::FromGraph("m")),
        ("n", AlgoParam::FromGraph("n")),
        ("k", AlgoParam::FromGraph("k")),
        ("mr", AlgoParam::FromDeviceProfile("gemm_mr")),
        ("nr", AlgoParam::FromDeviceProfile("gemm_nr")),
        ("k_step", AlgoParam::FromDeviceProfile("simd_lanes")),
    ],
    micro_kernel: Some(&MICRO_KERNEL_NAIVE),
};

// GEMM_GPU_TILED, GEMM_GPU_PIPELINED, GEMM_AMX_TILE 同理
```

### §4.3 Norm 模板示例

```rust
// algo_templates/norm.rs

pub static NORM_RMS: AlgoTemplate = AlgoTemplate {
    name: "NormRms",
    strategy: AlgoStrategy::NormRms,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Phase 1: reduce — x² 累加
        AlgoStep::Loop { bound: "dim", step: "simd_lanes", body: &[
            AlgoStep::TraceBody(&[
                // x² → acc (SymExec 可自动提取的逐元素 trace)
            ]),
        ]},
        AlgoStep::Reduce { op: ReduceOp::Sum },
        // Phase 2: finalize — rsqrt(sum/n)
        AlgoStep::TraceBody(&[
            // 1/sqrt(mean) → scale
        ]),
        // Phase 3: transform — x * scale
        AlgoStep::Loop { bound: "dim", step: "simd_lanes", body: &[
            AlgoStep::TraceBody(&[
                // x * scale → output
            ]),
        ]},
    ],
    params: &[
        ("dim", AlgoParam::FromGraph("hidden_dim")),
        ("simd_lanes", AlgoParam::FromDeviceProfile("simd_lanes")),
    ],
    micro_kernel: None,
};
```

### §4.4 模板注册表

```rust
// algo_templates/mod.rs

pub static GEMM_TEMPLATES: &[AlgoTemplate] = &[
    &GEMM_NAIVE, &GEMM_BLIS, &GEMM_AMX_TILE,
    &GEMM_GPU_TILED, &GEMM_GPU_PIPELINED,
];

pub fn select_template(
    strategy: AlgoStrategy,
    device: &DeviceProfile,
) -> Result<&'static AlgoTemplate, CompilerError> {
    let candidates = match strategy_family(strategy) {
        StrategyFamily::Gemm => GEMM_TEMPLATES,
        StrategyFamily::Attn => ATTN_TEMPLATES,
        StrategyFamily::Norm => NORM_TEMPLATES,
        // ...
    };
    candidates.iter()
        .filter(|t| t.strategy == strategy && device.satisfies(t.device_req))
        .max_by_key(|t| t.device_req.priority())
        .ok_or_else(|| CompilerError::Unsupported(
            format!("no template for {:?} on {:?}", strategy, device)
        ))
}
```

## §5 归一管线全貌

### §5.1 完整数据流

```
CompilerGraph (op + dimensions + tensor metadata)
  │
  ├─ OpClass::Elementwise ──────────────────────┐
  │   Scalar fn → SymExec → Vec<TraceOp> ────────┤
  │                                              │
  ├─ OpClass::NormLike ─────────────────────────┤
  │   AlgoTemplate(NORM_RMS) → Vec<TraceOp> ────┤
  │   (TraceBody 内的逐元素 trace 来自 SymExec)   │
  │                                              │
  ├─ OpClass::Gemm ─────────────────────────────┤
  │   AlgoTemplate(GEMM_BLIS) → Vec<TraceOp> ───┤
  │   (MicroKernel 内的 FMA 来自 SymExec)         │
  │                                              │
  ├─ OpClass::Injective (RoPE) ─────────────────┤
  │   AlgoTemplate(ROPE_STANDARD) → Vec<TraceOp>─┤
  │                                              │
  ├─ OpClass::Gather (QuantGather) ─────────────┤
  │   AlgoTemplate(EMBED) or SymExec → TraceOp ──┤
  │                                              │
  └─ OpClass::Moe ──────────────────────────────┤
      AlgoTemplate(MOE_ROUTER) → Vec<TraceOp> ──┘
                                                    │
                                                    ▼
                                            auto_lower_trace()
                                                    │
                                                    ▼
                                            Vec<VmInstr>
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                              x86_lower      aarch64_lower     gpu_lower
                              vmovups        ldr q            ld.global
                              vfmadd231      fmla             mma.sync
                                    │               │               │
                                    └───────────────┼───────────────┘
                                                    ▼
                                              机器码 (纯执行)
```

### §5.2 TraceOp 来源归一

| 来源 | 产出 | 走哪条路 |
|------|------|---------|
| `SymExec::trace(scalar_fn)` | `Vec<TraceOp>` | `auto_lower_trace()` |
| `instantiate_template(template)` | `Vec<TraceOp>` | `auto_lower_trace()` |
| 测试手动构造 | `Vec<TraceOp>` | `auto_lower_trace()` |

**三种来源汇聚到同一个 `auto_lower_trace()` 调用点。** auto_select 不知道也不关心 TraceOp 来自哪里——它只做查表映射。

### §5.3 迁移路径

| 手写函数 | 替换方式 | 产出 |
|---------|---------|------|
| `emit_gemm_naive_inline` | `GEMM_NAIVE` → TraceOp | `auto_lower_trace()` |
| `emit_gemm_blis_inline` | `GEMM_BLIS` → TraceOp | `auto_lower_trace()` |
| `emit_gemm_gpu_tiled_inline` | `GEMM_GPU_TILED` → TraceOp | `auto_lower_trace()` |
| `emit_gemm_gpu_pipelined` | `GEMM_GPU_PIPELINED` → TraceOp | `auto_lower_trace()` |
| `emit_normlike_inline` | `NORM_RMS/NORM_LAYER` → TraceOp | `auto_lower_trace()` |
| `emit_rope_inline` | `ROPE_STANDARD/PARTIAL` → TraceOp | `auto_lower_trace()` |
| `emit_moe_router_gemv_inline` | `MOE_ROUTER_TOPK` → TraceOp | `auto_lower_trace()` |
| `emit_moe_topk_dispatch_inline` | `MOE_TOPK` → TraceOp | `auto_lower_trace()` |
| `emit_moe_packed_inline` | `MOE_PACKED` → TraceOp | `auto_lower_trace()` |

**所有替换遵循同一模式**：选择模板 → 实例化为 TraceOp → `auto_lower_trace()`。plan_lower.rs 从 12,891 行手写代码变成模板选择+单行调用。

### §5.4 JIT 纯执行器验证

**JIT 从 auto_select 往下不知道任何算法信息**：

| 层 | 输入 | 输出 | 知道 "GEMM" 吗 |
|---|---|---|---|
| auto_select | `TraceOp::Fma(acc, a, b)` | `VmInstr::Fma { ... }` | 不知道 |
| auto_select | `TraceOp::PanelLoad { matrix: A }` | 多条 `VmInstr::VecLoad` | 不知道 |
| auto_select | `TraceOp::Loop { body }` | `LoopBegin` + body + `LoopEnd` | 不知道 |
| x86_lower | `VmInstr::Fma` | `vfmadd231ps` | 不知道 |
| x86_lower | `VmInstr::VecLoad` | `vmovups` | 不知道 |
| aarch64_lower | `VmInstr::Fma` | `fmla` | 不知道 |
| gpu_lower | `VmInstr::TileMma` | `mma.sync` | 不知道 |

**ISA lowering 层从始至终不知道自己在编译什么算法。** 它只是机械地把 VmInstr 翻译成目标架构的机器码。

## §6 REQ 列表

### REQ-AT-001: AlgoTemplate 数据模型

定义 `AlgoTemplate`, `AlgoStep`, `AlgoParam`, `AlgoStrategy`, `DeviceReq`, `MicroKernelDef` 纯数据结构。零方法，零代码生成逻辑。

**文件**: `codegen/vm/algo_template.rs`（新）

### REQ-AT-002: TraceOp 结构型扩展

在 `trace.rs` 新增 TraceOp 变体: `Loop`, `PanelLoad`, `PanelStore`, `PackBuffer`, `SharedMemDeclare`, `AsyncCopyToShared`, `AsyncWaitGroup`, `SyncBarrier`, `TileConfig`, `TileMma`, `TileRelease`, `Softmax`。

**文件**: `codegen/vm/trace.rs`（修改）

### REQ-AT-003: auto_select 新增映射

在 `auto_select.rs` 为 REQ-AT-002 新增的每个 TraceOp 变体实现 `dispatch_trace_op` match arm。所有映射产 VmInstr，不绕过管线。

**文件**: `codegen/vm/auto_select.rs`（修改）

### REQ-AT-004: GEMM 模板库

定义 5 个 GEMM 模板: `GEMM_NAIVE`, `GEMM_BLIS`, `GEMM_AMX_TILE`, `GEMM_GPU_TILED`, `GEMM_GPU_PIPELINED`。

**文件**: `codegen/vm/algo_templates/gemm.rs`（新）

### REQ-AT-005: Attention + Norm + RoPE + MoE 模板库

定义 Attention (MHA/GQA/MLA)、Norm (RmsNorm/LayerNorm)、RoPE (Standard/Partial)、MoE (Router/TopK/Packed) 模板。

**文件**: `codegen/vm/algo_templates/attention.rs`, `norm.rs`, `rope.rs`, `moe.rs`（新）

### REQ-AT-006: 模板解释器

实现 `instantiate_template()`: 遍历 AlgoStep 树，翻译为 `Vec<TraceOp>`。每个 AlgoStep 映射到对应 TraceOp 变体。微核展开为 TraceOp::Loop + Fma 序列。

**文件**: `codegen/vm/algo_interpreter.rs`（新）

### REQ-AT-007: 模板注册表 + 策略选择

实现 `select_template()`, `DeviceReq::priority()`, `DeviceProfile::satisfies()`。

**文件**: `codegen/vm/algo_templates/mod.rs`（新）

### REQ-AT-008: GEMM emit 函数迁移

`emit_gemm_inline_with_hook` → `select_template()` + `instantiate_template()` + `auto_lower_trace()`。删除 5 个手写 GEMM emit 函数。

**文件**: `codegen/vm/plan_lower.rs`

### REQ-AT-009: Norm/RoPE/MoE emit 函数迁移

`emit_normlike_inline`, `emit_rope_inline`, `emit_moe_*_inline` → 模板解释器 + `auto_lower_trace()`。

**文件**: `codegen/vm/plan_lower.rs`

### REQ-AT-010: Sampling 模板 + 迁移

定义采样管线模板 (Argmax → TemperatureScale → Softmax → TopK → TopP → Multinomial)。迁移手写采样 VmInstr 序列。

**文件**: `codegen/vm/algo_templates/sampling.rs`（新）, `plan_lower.rs`

### REQ-AT-011: 验证 — 数值对齐

所有迁移后通过 `cargo test --lib` + E2E 测试验证数值一致。模板产出与原手写 emit 函数语义等价。

## §7 实施顺序

```
REQ-AT-001 (数据模型)
    ↓
REQ-AT-002 + 003 (TraceOp 扩展 + auto_select 映射 — 可并行)
    ↓
REQ-AT-004 + 005 (模板库 — 可并行)
    ↓
REQ-AT-006 (解释器) + REQ-AT-007 (注册表 — 可并行)
    ↓
REQ-AT-008 + 009 (GEMM + 其他迁移 — 可并行)
    ↓
REQ-AT-010 (Sampling)
    ↓
REQ-AT-011 (验证)
```

## §8 文件结构

```
gllm-kernels/src/compiler/
├── trace.rs                          — [REQ-AT-002] TraceOp 结构型扩展
├── codegen/vm/
│   ├── algo_template.rs              — [REQ-AT-001] 数据模型
│   ├── algo_interpreter.rs           — [REQ-AT-006] 解释器 (产 TraceOp)
│   ├── auto_select.rs                — [REQ-AT-003] 新增 TraceOp→VmInstr 映射
│   ├── algo_templates/               — [REQ-AT-004,005,010] 模板库 (纯数据)
│   │   ├── mod.rs                    — [REQ-AT-007] 注册表 + 选择
│   │   ├── gemm.rs                   — GEMM 5 模板
│   │   ├── attention.rs              — Attention 3 模板
│   │   ├── norm.rs                   — Norm 2 模板
│   │   ├── rope.rs                   — RoPE 2 模板
│   │   ├── moe.rs                    — MoE 2 模板
│   │   └── sampling.rs              — Sampling 模板
│   ├── plan_lower.rs                 — [REQ-AT-008,009,010] 大幅瘦身
│   └── ...
```

**归一验证**：`grep -rn "auto_lower_trace" plan_lower.rs` — 所有算法调用点必须经过这一个函数。
