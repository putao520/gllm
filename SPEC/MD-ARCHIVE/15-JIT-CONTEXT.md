# 统一 JIT 编译上下文 — 硬件资源生命周期追踪

> **SSOT**: 本文件定义 `JitContext`——JIT 编译过程中的统一硬件资源管理器。
> 设备能力定义见 `SPEC/02-HARDWARE.md`。
> 编译管线架构见 `SPEC/01-JIT-PIPELINE.md`。
> 寄存器分配器见 `isa_profile.rs` / `reg_alloc.rs`。
> 核心铁律见 `SPEC/00-PHILOSOPHY.md`。

## 1. 问题与目标

### 1.1 现状缺陷

JIT 编译 Phase 3（ISA Lowering）期间，各 Lowerer 对硬件资源的使用是"盲"的：

| 资源类型 | X86Lower | AArch64Lower | GpuLower |
|----------|----------|-------------|----------|
| GPR | IsaProfile 定义，RegAllocator 分配 | IsaProfile 定义 | 无（PTX 虚拟寄存器） |
| SIMD/Vec | IsaProfile 定义，RegAllocator 分配 | IsaProfile 定义 | 无（PTX 虚拟寄存器） |
| Tile (AMX/SME) | `amx_tile_dtype: Option<DType>` 单值 | `tile_dtype: Option<DType>` 单值 | `tile_rows/tile_cols` + `tmem_allocated: bool` |
| SMEM | 不适用 | 不适用 | prologue 声明总量，emit 期间无追踪 |
| TMEM | 不适用 | 不适用 | `tmem_allocated: bool`，无 offset/size |
| Stack | `spill_base_off`，无峰值查询 | 无 | 不适用 |
| Barrier | 不适用 | 不适用 | 隐式在 TileConfig/TileMma 中，无独立追踪 |

**核心问题**：

1. **无法回答"此时此刻设备各部分使用多少"** — 没有 per-instruction 资源快照
2. **跨设备类型不统一** — CPU 用 bool，GPU 用 bool，Tile 用 Option<DType>，SMEM 无追踪
3. **预算检查分散** — SMEM 超用/Tile 冲突/Stack 溢出只能在运行时崩溃发现
4. **优化决策缺乏依据** — 融合深度/epilogue 策略无法精确查询资源余量

### 1.2 设计目标

| 目标 | 度量 |
|------|------|
| **统一抽象** | 一套 `ResourceKind` 枚举覆盖 CPU/GPU/NPU 所有资源类型 |
| **生命周期追踪** | 每个资源实例的 alloc → live → release 全程跟踪 |
| **峰值查询** | 任意时刻可查询 "截至目前 GPR 峰值/SMEM 峰值/Tile 峰值" |
| **预算门控** | 资源分配前自动检查是否超出设备物理限制 |
| **零运行时开销** | JitContext 仅在编译时存在，不进入推理热路径 |
| **向后兼容** | 现有 Lowerer 逐步接入，不破坏已有功能 |

## 2. 统一资源模型

### 2.1 ResourceKind

覆盖所有设备类型的资源分类。每个 ResourceKind 对应一组物理实例，由 `ResourceBudget` 定义数量上限。

```rust
/// 硬件资源种类 — 统一覆盖 CPU/GPU/NPU。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    // ── 通用寄存器 ──
    /// 通用寄存器: x86 R0-R31(APX), ARM X0-X30, GPU R0-R255
    Gpr,
    /// SIMD/向量寄存器: x86 YMM/ZMM, ARM V0-V31 / SVE Z0-Z31, GPU VGPR, Apple SIMDgroup
    SimdVec,
    /// 掩码/谓词寄存器: x86 k0-k7, ARM SVE P0-P15, GPU predicate %p0-%p7
    Predicate,

    // ── 加速器 Tile ──
    /// Tile 寄存器: x86 AMX TMM0-7, ARM SME ZA(2D array), GPU TMEM
    Tile,
    /// Tile 累加器 (逻辑概念): GPU WGMMMA/tcgen05 fragment accumulator
    /// 不对应独立物理寄存器，但占用寄存器文件预算
    TileAccumulator,

    // ── 内存层级 ──
    /// CPU 栈帧 (spill slots + callee-save + ABI args)
    Stack,
    /// GPU 共享内存: NVIDIA SMEM / AMD LDS / Metal Threadgroup
    SharedMem,
    /// GPU 张量内存: SM100+ TMEM (256KB/SM)
    TensorMem,

    // ── 同步原语 ──
    /// GPU mbarrier (SM90+) / CPU fence
    Barrier,
}
```

### 2.2 ResourceKind 与设备映射

| ResourceKind | x86_64 | AArch64 | NVIDIA GPU | AMD GPU | Apple GPU |
|-------------|--------|---------|-----------|---------|-----------|
| Gpr | RAX-R15 (APX: R0-R31) | X0-X30, SP, LR | R0-R255 (PTX) | VGPR pool | — |
| SimdVec | YMM0-15 / ZMM0-31 | V0-V31 / Z0-Z31 | — (implicit) | — (implicit) | simdgroup float8x8 |
| Predicate | k0-k7 (AVX-512) | P0-P15 (SVE) | %p0-%p7 | — | — |
| Tile | AMX TMM0-7 | SME ZA (VL×VL) | TMEM (SM100+) | — | — |
| TileAccumulator | — | — | WGMMA fragment | MFMA acc | — |
| Stack | SysV stack frame | AAPCS64 stack | — | — | — |
| SharedMem | — | — | SMEM per block | LDS per CU | Threadgroup |
| TensorMem | — | — | TMEM (SM100+) | — | — |
| Barrier | — | — | mbarrier (SM90+) | — | — |

### 2.3 资源生命周期状态

```rust
/// 单个资源实例的生命周期状态。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceState {
    /// 未分配，可供使用
    Free,
    /// 已分配，正在使用中
    /// `purpose` 用于审计日志 (如 "loop_counter", "tcgen05_acc")
    /// `alloc_instr` 是分配时 VmProgram 中的指令索引
    Live {
        purpose: &'static str,
        alloc_instr: usize,
    },
}
```

**设计决策**：不区分 Allocated/Live/Released 三状态——资源要么 Free 要么 Live。
JitContext 通过时间线（timeline）记录历史分配/释放事件，而非在 ResourceState 上堆叠状态。
这样 `peak_usage` 的计算只需遍历时间线，不需要复杂的状态机。

## 3. ResourceBudget — 设备资源上限

ResourceBudget 从 `IsaProfile` 一次性派生，在编译期间不可变。它是资源分配的门控阈值。

```rust
/// 设备物理资源上限 — 从 IsaProfile 派生的不可变配置。
#[derive(Debug, Clone)]
pub struct ResourceBudget {
    pub gpr_total: usize,
    pub simd_vec_total: usize,
    pub predicate_total: usize,
    pub tile_total: usize,
    pub tile_accumulator_total: usize,
    pub stack_bytes: usize,
    pub shared_mem_bytes: usize,
    pub tensor_mem_bytes: usize,
    pub barrier_total: usize,
}
```

### 3.1 从 IsaProfile 派生

```rust
impl ResourceBudget {
    pub fn from_isa_profile(profile: &IsaProfile) -> Self {
        Self {
            gpr_total: profile.gpr_regs.len(),
            simd_vec_total: profile.vec_regs.len(),
            predicate_total: profile.mask_regs.len(),
            tile_total: profile.tile_regs.len(),
            tile_accumulator_total: match &profile.platform {
                Platform::Cuda { sm_version, reg_file_per_sm, .. } if *sm_version >= 90 => {
                    // WGMMA: 每个 warpgroup 4×WGMMA accumulator fragments
                    4
                }
                _ => 0,
            },
            stack_bytes: 4096, // 默认 4KB，StackFrame 计算后更新
            shared_mem_bytes: match &profile.platform {
                Platform::Cuda { shared_mem_kb, .. } => shared_mem_kb * 1024,
                Platform::Hip { lds_size_kb, .. } => lds_size_kb * 1024,
                Platform::Metal { threadgroup_mem_kb, .. } => threadgroup_mem_kb * 1024,
                _ => 0,
            },
            tensor_mem_bytes: match &profile.platform {
                Platform::Cuda { tmem_size_kb, .. } => tmem_size_kb * 1024,
                _ => 0,
            },
            barrier_total: match &profile.platform {
                Platform::Cuda { has_warp_spec, .. } if *has_warp_spec => 2,
                _ => 0,
            },
        }
    }
}
```

### 3.2 动态更新

StackFrame 计算完成后，通过 `JitContext::update_stack_budget(bytes)` 更新栈上限。
SMEM scratch 声明后，通过 `JitContext::update_smem_used(bytes)` 扣减可用量。

## 4. ResourceEvent — 资源事件时间线

```rust
/// 资源生命周期事件 — 追踪每次分配和释放。
#[derive(Debug, Clone)]
pub struct ResourceEvent {
    /// 事件发生的 VmProgram 指令索引
    pub instr_idx: usize,
    /// 资源种类
    pub kind: ResourceKind,
    /// 资源实例索引 (在 kind 池中的位置)
    pub instance: usize,
    /// 事件类型
    pub event_type: ResourceEventType,
    /// 分配用途 (用于审计)
    pub purpose: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceEventType {
    Allocate,
    Release,
}
```

## 5. JitContext — 核心结构

```rust
/// JIT 编译上下文 — Phase 3 期间的统一硬件资源管理器。
///
/// ## 生命周期
/// 1. 创建: `JitContext::new(isa_profile)` — 在 Phase 3 开始前创建
/// 2. 使用: Lowerer 通过 `allocate`/`release` 追踪资源
/// 3. 查询: 任意时刻可查询 `peak`/`available`/`snapshot`
/// 4. 销毁: Phase 3 结束后丢弃，不进入推理运行时
///
/// ## 线程安全
/// JitContext 是 !Sync + !Send — 仅在单线程编译期间使用。
/// 推理运行时完全不需要 JitContext。
pub struct JitContext {
    // ── 设备能力 (immutable) ──
    budget: ResourceBudget,
    profile: HardwareProfile,

    // ── 资源池状态 (mutable) ──
    /// 每种资源的实例状态向量。
    /// resources[ResourceKind::Gpr] = Vec<ResourceState> 长度 = budget.gpr_total
    /// resources[ResourceKind::SimdVec] = Vec<ResourceState> 长度 = budget.simd_vec_total
    resources: HashMap<ResourceKind, Vec<ResourceState>>,

    // ── 事件时间线 (append-only) ──
    events: Vec<ResourceEvent>,

    // ── 峰值追踪 (单调递增) ──
    peak_live: HashMap<ResourceKind, usize>,

    // ── 内存类资源用量追踪 ──
    stack_used: usize,
    smem_used: usize,
    tmem_used: usize,

    // ── 当前指令索引 (由 Lowerer 推进) ──
    current_instr: usize,
}
```

### 5.1 核心方法

```rust
impl JitContext {
    /// 从 IsaProfile 创建 JitContext。
    pub fn new(profile: &IsaProfile) -> Self;

    // ── 资源分配 ──

    /// 分配一个指定类型的资源实例。
    /// 返回实例索引 (在 kind 池中的位置)。
    /// 如果资源池耗尽，返回 Err(含当前峰值和上限信息)。
    pub fn allocate(&mut self, kind: ResourceKind, purpose: &'static str)
        -> Result<usize, ResourceExhausted>;

    /// 释放一个资源实例。
    pub fn release(&mut self, kind: ResourceKind, instance: usize);

    /// 预分配 N 个连续实例 (用于 SMEM/TMEM 区域分配)。
    /// 返回起始索引。
    pub fn allocate_region(
        &mut self, kind: ResourceKind, count: usize, purpose: &'static str,
    ) -> Result<usize, ResourceExhausted>;

    // ── 查询接口 ──

    /// 当前指定资源种类的活跃实例数。
    pub fn live_count(&self, kind: ResourceKind) -> usize;

    /// 指定资源种类的历史峰值活跃实例数。
    pub fn peak(&self, kind: ResourceKind) -> usize;

    /// 指定资源种类当前可用实例数。
    pub fn available(&self, kind: ResourceKind) -> usize;

    /// 指定资源种类的物理上限。
    pub fn capacity(&self, kind: ResourceKind) -> usize;

    /// 内存类资源的已用字节数。
    pub fn mem_used(&self, kind: ResourceKind) -> usize;

    /// 内存类资源的可用字节数。
    pub fn mem_available(&self, kind: ResourceKind) -> usize;

    // ── 快照 ──

    /// 当前时刻的资源使用快照。
    pub fn snapshot(&self) -> ResourceSnapshot;

    // ── 指令推进 ──

    /// 推进当前指令索引。Lowerer 在 emit 每条 VmInstr 前调用。
    pub fn advance_to(&mut self, instr_idx: usize);

    // ── 动态预算更新 ──

    /// 更新栈预算 (StackFrame 计算完成后调用)。
    pub fn update_stack_budget(&mut self, bytes: usize);

    /// 声明 SMEM 使用量 (prologue 声明 shared memory 后调用)。
    pub fn declare_smem_usage(&mut self, bytes: usize);

    /// 声明 TMEM 使用量 (tcgen05.alloc 后调用)。
    pub fn declare_tmem_usage(&mut self, bytes: usize);
}
```

### 5.2 ResourceSnapshot — 时间点快照

```rust
/// 某一时刻的资源使用快照。
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// 快照时的 VmProgram 指令索引
    pub instr_idx: usize,
    /// 每种资源的当前活跃数
    pub live: HashMap<ResourceKind, usize>,
    /// 每种资源的历史峰值
    pub peak: HashMap<ResourceKind, usize>,
    /// 每种资源的可用余量
    pub available: HashMap<ResourceKind, usize>,
    /// 内存类资源已用量 (字节)
    pub mem_used: HashMap<ResourceKind, usize>,
}
```

### 5.3 ResourceExhausted — 资源耗尽错误

```rust
/// 资源耗尽错误 — 包含诊断信息帮助优化决策。
#[derive(Debug, Clone)]
pub struct ResourceExhausted {
    pub kind: ResourceKind,
    pub requested: usize,
    pub capacity: usize,
    pub current_live: usize,
    pub peak: usize,
    pub suggestion: &'static str,
}
```

## 6. 与现有管线的集成

### 6.1 管线位置

```
Phase 0: Scalar → Phase 1: SymExec → Phase 2: Fusion
    ↓
Phase 2.5: StrategySelector (使用 JitContext.budget 查询资源余量)
    ↓
Phase 3: ISA Lowering (使用 JitContext 追踪资源生命周期)
    ├── RegAllocator: 消费 JitContext.budget 作为分配约束
    ├── StackFrame: 消费 JitContext.stack_used 计算栈帧布局
    ├── X86Lower: allocate(Tile) / allocate(Stack) / release(Tile)
    ├── AArch64Lower: allocate(Tile) / allocate(Stack) / release(Tile)
    └── GpuLower: allocate(Tile) / declare_smem / declare_tmem / allocate(Barrier)
```

### 6.2 与 RegAllocator 的关系

RegAllocator 负责 VReg → PhysReg 映射，是 GPR/SimdVec/Predicate/Tile 的**静态分配器**。
JitContext 是**动态追踪器**，记录资源在 emit 期间的使用时序。

**协作方式**：

```
                    ┌─ IsaProfile ─┐
                    │              │
                    ▼              ▼
              JitContext      RegAllocator
              (预算门控)       (VReg→PhysReg)
                    │              │
                    ▼              ▼
         allocate 检查余量    compute_intervals
         release 更新状态     → alloc (映射)
                    │              │
                    └──────┬───────┘
                           ▼
                      ISA Lowerer
                  (emit 时查询双方)
```

具体集成：
- RegAllocator 分配后，Lowerer 调用 `JitContext::allocate(Gpr, purpose)` 登记使用
- RegAllocator 释放后，Lowerer 调用 `JitContext::release(Gpr, instance)` 登记释放
- 对于 SMEM/TMEM/Barrier 等 RegAllocator 不管理的资源，JitContext 是唯一的追踪器

### 6.3 与 StackFrame 的关系

StackFrame 计算 callee-save + ABI args + spill slots 的布局。
JitContext 追踪 stack_used 峰值，与 StackFrame 的计算结果互相验证：

```rust
// StackFrame 计算完成后
let stack_size = stack_frame.total_bytes();
ctx.update_stack_budget(stack_size);

// emit 期间如果 stack 动态增长
ctx.allocate(Stack, "temp_buffer")?; // 检查不超出预算
```

### 6.4 Lowerer 接入方式

#### X86Lower 接入

```rust
pub struct X86Lower {
    // ── 现有字段保持 ──
    use_avx512: bool,
    // ...

    // ── 新增 ──
    ctx: JitContext,  // 替代分散的 bool/Option 追踪
}

impl X86Lower {
    fn emit_tile_config(&mut self, ...) {
        // 替代隐式 amx_tile_dtype = Some(dtype)
        self.ctx.allocate(ResourceKind::Tile, "amx_tmm0")?;
        self.ctx.allocate(ResourceKind::Tile, "amx_tmm1")?;
        // ...
    }

    fn emit_tile_release(&mut self, ...) {
        // 替代隐式 amx_tile_dtype = None
        self.ctx.release(ResourceKind::Tile, 0);
        self.ctx.release(ResourceKind::Tile, 1);
    }
}
```

#### GpuLower 接入

```rust
pub struct GpuLower {
    // ── 现有字段保持 ──
    dialect: GpuDialect,
    // ...

    // ── 新增 ──
    ctx: JitContext,  // 替代 tmem_allocated: bool
}

impl GpuLower {
    fn emit_tile_config_sm100(&mut self, ...) {
        // 替代隐式 tmem_allocated = true
        let tmem_bytes = tile_cols * 4 * 2; // 2-CTA 模式翻倍
        self.ctx.declare_tmem_usage(tmem_bytes);
        self.ctx.allocate(ResourceKind::Tile, "tcgen05_acc")?;
    }

    fn emit_smem_scratch(&mut self, size: usize) {
        // 新增：SMEM 用量追踪
        self.ctx.declare_smem_usage(size);
        // 检查不超出 SMEM 预算
        if self.ctx.mem_available(ResourceKind::SharedMem) < needed {
            return Err("SMEM budget exceeded");
        }
    }

    fn emit_tile_release_sm100(&mut self, ...) {
        self.ctx.release(ResourceKind::Tile, 0);
    }
}
```

## 7. 资源预算与硬件映射

### 7.1 CPU x86_64 预算

| ResourceKind | AVX2 | AVX-512 | AVX10.2 | AMX |
|-------------|------|---------|---------|-----|
| Gpr | 10 (16-6) | 10 | 25 (31-6) | 10 |
| SimdVec | 10 (16-6) | 26 (32-6) | 26 | 26 |
| Predicate | 0 | 8 | 8 | 8 |
| Tile | 0 | 0 | 0 | 8 (TMM0-7) |
| Stack | 4096+ | 4096+ | 4096+ | 4096+ |

注：GPR/SimdVec 数量 = 物理总数 - scratch 保留数。

### 7.2 ARM AArch64 预算

| ResourceKind | NEON | SVE2 | SME2 |
|-------------|------|------|------|
| Gpr | 28 (X0-X30 - SP/LR/ZR) | 28 | 28 |
| SimdVec | 32 (V0-V31) | 32 (Z0-Z31) | 32 |
| Predicate | 0 | 16 (P0-P15) | 16 |
| Tile | 0 | 0 | 1 (ZA array) |
| Stack | 4096+ | 4096+ | 4096+ |

### 7.3 NVIDIA GPU 预算

| ResourceKind | SM80 | SM90 | SM100 |
|-------------|------|------|-------|
| Gpr | 255 (PTX virtual) | 255 | 255 |
| SimdVec | 0 (implicit in WGMMA) | 0 | 0 |
| Predicate | 8 (%p0-%p7) | 8 | 8 |
| Tile | 0 | 0 | 1 (TMEM) |
| TileAccumulator | 0 | 4 (WGMMA fragments) | 4 (tcgen05) |
| SharedMem | 48-164 KB | 48-228 KB | 48-228 KB |
| TensorMem | 0 | 0 | 256 KB |
| Barrier | 0 | 2 (mbarrier) | 2 |

### 7.4 AMD GPU 预算

| ResourceKind | CDNA2 (gfx908) | CDNA3 (gfx942) | CDNA4 (gfx950) |
|-------------|----------------|----------------|----------------|
| Gpr | 255 (virtual) | 255 | 255 |
| SimdVec | 0 (implicit in MFMA) | 0 | 0 |
| TileAccumulator | 4 (MFMA acc) | 4 | 4 |
| SharedMem | 64 KB (LDS) | 64 KB | 64 KB |

## 8. 融合决策增强

JitContext 使融合决策从"静态规则匹配"进化为"动态资源查询"。

### 8.1 Phase 2 融合 — 资源余量查询

```rust
// 融合决策中使用 JitContext 查询 SMEM 余量
fn can_fuse_epilogue(ctx: &JitContext, epilogue_ops: &[OpId]) -> bool {
    let smem_budget = ctx.mem_available(ResourceKind::SharedMem);
    let needed = estimate_epilogue_smem(epilogue_ops);
    smem_budget >= needed
}
```

### 8.2 Phase 2.5 StrategySelector — Tile 预算检查

```rust
// 选择 TileLevelFusion 前检查 Tile 资源余量
fn select_tile_fusion(ctx: &JitContext) -> bool {
    ctx.available(ResourceKind::Tile) > 0
        && ctx.mem_available(ResourceKind::SharedMem) > 0
}
```

### 8.3 Phase 3 Emit — 动态预算门控

```rust
// emit SMEM scratch 前检查余量
fn emit_smem_epilogue(ctx: &mut JitContext, ops: &[TraceOp]) -> Result<()> {
    let needed = estimate_scratch_bytes(ops);
    if ctx.mem_available(ResourceKind::SharedMem) < needed {
        return Err(ResourceExhausted {
            kind: ResourceKind::SharedMem,
            requested: needed,
            capacity: ctx.capacity(ResourceKind::SharedMem),
            current_live: ctx.live_count(ResourceKind::SharedMem),
            peak: ctx.peak(ResourceKind::SharedMem),
            suggestion: "reduce epilogue chain length or tile size",
        });
    }
    ctx.declare_smem_usage(needed);
    // ... emit epilogue ...
    Ok(())
}
```

## 9. 审计与诊断

### 9.1 资源使用报告

编译完成后，JitContext 可生成资源使用报告：

```rust
impl JitContext {
    /// 编译完成后的资源使用报告。
    pub fn usage_report(&self) -> ResourceReport {
        ResourceReport {
            peak_usage: self.peak_live.clone(),
            capacity: self.budget.clone(),
            utilization: self.peak_live.iter().map(|(k, &v)| {
                (*k, v as f64 / self.capacity(*k) as f64)
            }).collect(),
            warnings: self.generate_warnings(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceReport {
    pub peak_usage: HashMap<ResourceKind, usize>,
    pub capacity: ResourceBudget,
    pub utilization: HashMap<ResourceKind, f64>,
    pub warnings: Vec<ResourceWarning>,
}

#[derive(Debug, Clone)]
pub enum ResourceWarning {
    /// 资源利用率超过 90%
    NearExhaustion { kind: ResourceKind, peak: usize, capacity: usize },
    /// Tile 分配后未释放
    Leak { kind: ResourceKind, purpose: &'static str },
    /// 内存类资源使用不对称
    AsymmetricUsage { kind: ResourceKind, peak_pct: f64, avg_pct: f64 },
}
```

### 9.2 编译日志

当环境变量 `GLLM_DEBUG_RESOURCE` 设置时，JitContext 输出每次 allocate/release：

```
[JitContext] instr=42 allocate Tile:0 "amx_tmm0" (live=1, peak=1, cap=8)
[JitContext] instr=42 allocate Tile:1 "amx_tmm1" (live=2, peak=2, cap=8)
[JitContext] instr=58 release Tile:0 (live=1, peak=2, cap=8)
[JitContext] instr=58 release Tile:1 (live=0, peak=2, cap=8)
[JitContext] instr=100 declare_smem 8192 (used=8192/163840, 5.0%)
[JitContext] instr=120 declare_tmem 4096 (used=4096/262144, 1.6%)
```

## 10. 实现约束

### 10.1 性能约束

| 约束 | 目标 |
|------|------|
| `allocate()` 延迟 | ≤ 100ns (线性扫描 Free 列表) |
| `peak()` 查询 | O(1) (预计算) |
| `snapshot()` 创建 | ≤ 1μs (复制 9 个 usize + 3 个 HashMap entry) |
| 内存占用 | ≤ 64KB (典型: 9 种 × ~256 实例 × 16B) |
| 事件时间线 | ≤ 10K events (典型编译: ~500 VmInstr × ~5 resources) |

### 10.2 正确性约束

- **资源泄漏检测**: 编译结束时所有资源必须回到 Free 状态（Tile/SMEM/TMEM 除外，这些由硬件管理）
- **双重释放检测**: `release` 已 Free 的资源 → panic（编译时错误，不进入运行时）
- **超额分配检测**: `allocate` 超出 budget → 返回 Err
- **指令单调性**: `advance_to` 的 instr_idx 必须单调递增

### 10.3 编译时单态化

JitContext 在编译时创建、使用、销毁。**推理运行时零引用**：
- 不出现在 MegaKernel 的 ABI 参数中
- 不出现在 StackFrame 的运行时结构中
- 不出现在 Mega-Kernel CALL 的执行路径中

## 11. REQ 清单

### 11.1 核心 REQ

| ID | 描述 | 验收标准 |
|----|------|---------|
| REQ-JCTX-001 | ResourceKind 枚举覆盖所有设备类型 | 9 种 ResourceKind，x86_64/AArch64/CUDA/HIP/Metal 全映射 |
| REQ-JCTX-002 | ResourceBudget 从 IsaProfile 正确派生 | 每种硬件配置的 budget 与 SPEC §7 数值一致 |
| REQ-JCTX-003 | JitContext allocate/release 生命周期追踪 | Tile: alloc→live→release→free 全程可追踪 |
| REQ-JCTX-004 | 峰值查询接口 | peak(Gpr) 返回 GPR 历史最大同时活跃数 |
| REQ-JCTX-005 | 内存类资源用量追踪 | SMEM/TMEM/Stack 的已用量和可用量准确 |
| REQ-JCTX-006 | 预算门控 | 超出 capacity 的 allocate 返回 Err 而非 panic |
| REQ-JCTX-007 | ResourceReport 诊断报告 | 编译后可输出 peak/capacity/utilization/warnings |

### 11.2 集成 REQ

| ID | 描述 | 验收标准 |
|----|------|---------|
| REQ-JCTX-010 | X86Lower 接入 JitContext | AMX Tile alloc/release 通过 JitContext 追踪 |
| REQ-JCTX-011 | AArch64Lower 接入 JitContext | SME ZA Tile alloc/release 通过 JitContext 追踪 |
| REQ-JCTX-012 | GpuLower 接入 JitContext | TMEM/SMEM/Barrier 通过 JitContext 追踪 |
| REQ-JCTX-013 | RegAllocator 消费 JitContext.budget | 分配约束从 JitContext 读取而非独立计算 |
| REQ-JCTX-014 | 融合决策使用 JitContext 查询 | SMEM 余量检查通过 mem_available(SharedMem) |

### 11.3 质量 REQ

| ID | 描述 | 验收标准 |
|----|------|---------|
| REQ-JCTX-020 | 推理运行时零引用 | grep "JitContext" 在 executor/ 运行时代码中无结果 |
| REQ-JCTX-021 | 双重释放检测 | release 已 Free 资源 → 编译时 panic |
| REQ-JCTX-022 | 资源泄漏检测 | 编译结束检查 Tile/Barrier 已全部释放 |
| REQ-JCTX-023 | 所有硬件 Profile 测试 | 12 种 HardwareProfile × 9 种 ResourceKind = 108 组预算测试 |

## 12. 文件布局

```
gllm-kernels/src/compiler/
├── jit_context.rs        # JitContext + ResourceKind + ResourceBudget + ResourceSnapshot
├── hardware_profile.rs   # 现有，新增 resource_budget() → ResourceBudget 方法
├── codegen/vm/
│   ├── isa_profile.rs    # 现有，新增 into_jit_context() 方法
│   ├── reg_alloc.rs      # 现有，消费 JitContext.budget 作为约束
│   ├── x86_lower.rs      # 现有，接入 JitContext 替代分散追踪
│   ├── aarch64_lower.rs  # 现有，接入 JitContext 替代分散追踪
│   └── gpu_lower.rs      # 现有，接入 JitContext 替代 tmem_allocated 等
```

## 13. 迁移路径

### Phase 1: 核心数据结构
1. 创建 `jit_context.rs`，定义 ResourceKind/ResourceBudget/ResourceSnapshot/JitContext
2. IsaProfile 新增 `into_jit_context()` 方法
3. HardwareProfile 新增 `resource_budget()` 方法
4. 单元测试覆盖所有 ResourceKind 预算计算

### Phase 2: GpuLower 接入
1. GpuLower 构造时创建 JitContext
2. TileConfig: `declare_tmem_usage` + `allocate(Tile)`
3. TileMma: `allocate(TileAccumulator)` + `allocate(Barrier)` (SM90)
4. TileRelease: `release(Tile)` + `release(Barrier)`
5. SMEM scratch: `declare_smem_usage`
6. 测试验证 TMEM/SMEM/Barrier 追踪正确

### Phase 3: X86Lower 接入
1. X86Lower 构造时创建 JitContext
2. AMX TileConfig: `allocate(Tile)`
3. AMX TileRelease: `release(Tile)`
4. Stack: `update_stack_budget` + `allocate(Stack)`
5. 测试验证 AMX Tile 追踪正确

### Phase 4: AArch64Lower 接入
1. AArch64Lower 构造时创建 JitContext
2. SME ZA: `allocate(Tile)`
3. Stack: 同 X86Lower
4. 测试验证 SME ZA 追踪正确

### Phase 5: 融合决策增强
1. Phase 2 融合 pass 使用 JitContext 查询资源余量
2. Phase 2.5 StrategySelector 使用 JitContext 检查可行性
3. 编译后 ResourceReport 输出
