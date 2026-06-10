# VM 状态机动态布局跟踪 (ARCH-VM-STATE-TRACKING)

> **SSOT 声明**: 本文档定义 REGISTER-VM 半虚拟化的核心机制——VM 状态机在代码生成过程中动态跟踪内存布局、栈帧、指针和寄存器，所有物理位置由 VM 计算输出，禁止人工预设常量。
>
> **废弃**: `executable.rs::abi_slots` 硬编码常量模块（`Stack(24)` 等）。
>
> **交叉引用**: `symdim-threading-protocol.md`（SymDim 穿透）、`01-JIT-PIPELINE.md` §5（PlatformBackend）、`08-EXECUTOR.md` §4.7（ABI 契约）

## 1. 设计原则

### 1.1 为什么叫"半虚拟化"

传统 JIT（如 V8/HotSpot）：人预设 ABI 常量，codegen 硬编码偏移。
全虚拟化（如 JVM bytecode）：虚拟机完全隐藏物理布局，但性能损失大。

**REGISTER-VM 半虚拟化**：
- VM 状态机在 code generation 过程中**实时跟踪**物理布局
- 每条 VmInstr 的发射会**更新 VM 状态**（栈指针、寄存器占用、参数位置）
- 需要物理位置时，从 VM 状态中**查询**，而非引用预设常量
- ISA Lower 阶段看到的是 VM 状态的**快照**，不是硬编码的数字

### 1.2 核心不变量

| 铁律 | 说明 |
|------|------|
| **ARCH-VM-NO-MAGIC-OFFSET** | 禁止在任何文件中硬编码栈偏移数字（如 16/24/32/40）。所有偏移由 VM 状态机计算。 |
| **ARCH-VM-PROLOGUE-TRACKS** | prologue 生成过程中，VM 状态机记录每个 push/sub 操作对栈指针的影响，更新参数偏移表。 |
| **ARCH-VM-QUERY-NOT-ASSUME** | 需要参数位置时，查询 VM 状态（`vm_state.arg_location("output")`），不引用常量。 |

## 2. VM 状态机

### 2.1 VmState 结构

```rust
/// VM 代码生成状态机。
///
/// 在 prologue → body → epilogue 的整个生成过程中持续更新。
/// 所有物理位置（参数偏移、寄存器映射、栈帧布局）从此结构查询。
pub struct VmState {
    /// 当前栈指针相对于 frame base (rbp) 的偏移
    /// prologue push rbp 后 = 0
    /// 每次 push → -8，每次 sub rsp,N → -N
    rsp_offset: i32,

    /// ABI 参数位置表——在 prologue 开始前由平台 ABI 规则初始化。
    /// key = 参数语义名称（"input"/"output"/"seq_len"/...）
    /// value = 相对 rbp 的字节偏移（prologue 后的位置）
    ///
    /// 初始化规则（x86 SysV）:
    ///   前 6 个整数/指针参数 → 寄存器（rdi/rsi/rdx/rcx/r8/r9）
    ///   第 7+ 个参数 → [rbp + 16 + (arg_idx - 6) * 8]
    ///                         ^^ 16 = ret_addr(8) + saved_rbp(8)
    ///
    /// 这个 16 不是硬编码常量——它是 `push rbp` 后的数学推导结果，
    /// 由 `init_for_platform()` 根据 ABI 规则计算。
    arg_locations: HashMap<String, ArgLocation>,

    /// 寄存器占用状态
    gpr_state: [GprState; 16],    // x86: rax-r15
    vec_state: [VecState; 32],    // x86: ymm0-ymm31 / zmm0-zmm31

    /// Callee-saved 寄存器的保存位置（push 顺序决定）
    callee_save_locations: Vec<(PhysGpr, i32)>,  // (寄存器, rbp-relative 偏移)

    /// 当前 spill 区域的起始偏移（相对 rbp，负值）
    spill_base: i32,
}

/// 参数的物理位置。
pub enum ArgLocation {
    /// 在寄存器中（前 6 个参数）
    Register(PhysGpr),
    /// 在栈上（第 7+ 个参数）
    /// offset 是相对 rbp 的正偏移（调用方的栈帧）
    Stack(i32),
}
```

### 2.2 初始化（平台 ABI 驱动）

```rust
impl VmState {
    /// 从平台 ABI 规则初始化参数位置表。
    ///
    /// x86 SysV: 前 6 个 → rdi/rsi/rdx/rcx/r8/r9
    /// 第 7+ 个 → [rbp + stack_arg_base + (idx - 6) * 8]
    /// stack_arg_base = 16（= ret_addr + saved_rbp，由 ABI 规范推导）
    pub fn init_x86_sysv(param_names: &[&str]) -> Self {
        let reg_args = [
            PhysGpr(7),  // rdi
            PhysGpr(6),  // rsi
            PhysGpr(2),  // rdx
            PhysGpr(1),  // rcx
            PhysGpr(8),  // r8
            PhysGpr(9),  // r9
        ];
        let stack_arg_base: i32 = 16; // ret_addr(8) + push_rbp(8) — ABI 数学推导

        let mut arg_locations = HashMap::new();
        for (i, &name) in param_names.iter().enumerate() {
            let loc = if i < 6 {
                ArgLocation::Register(reg_args[i])
            } else {
                ArgLocation::Stack(stack_arg_base + ((i - 6) as i32) * 8)
            };
            arg_locations.insert(name.to_string(), loc);
        }

        Self {
            rsp_offset: 0,
            arg_locations,
            gpr_state: [GprState::Free; 16],
            vec_state: [VecState::Free; 32],
            callee_save_locations: Vec::new(),
            spill_base: 0,
        }
    }
}
```

### 2.3 MegaKernelFn 参数名序列

```rust
/// MegaKernelFn 的参数名序列——与签名一一对应。
/// VmState::init_x86_sysv() 的输入。
pub const COMPILED_LAYER_PARAM_NAMES: &[&str] = &[
    "input",       // arg 0: rdi
    "weights",     // arg 1: rsi
    "kv_cache",    // arg 2: rdx
    "positions",   // arg 3: rcx
    "seq_lens",    // arg 4: r8
    "batch_size",  // arg 5: r9
    "seq_len",     // arg 6: [rbp+16]  — 由 ABI 推导，非硬编码
    "output",      // arg 7: [rbp+24]
    "scratchpad",  // arg 8: [rbp+32]
    "telemetry",   // arg 9: [rbp+40]
];
```

**注意**: `[rbp+16]` 不是硬编码——它是 `stack_arg_base(16) + (6-6)*8 = 16` 的计算结果。如果 ABI 变更（如 Windows x64 的 shadow space），`stack_arg_base` 会不同，所有偏移自动更新。

## 3. Prologue 状态跟踪

### 3.1 每条 prologue 指令更新 VM 状态

```rust
impl X86Lower {
    fn emit_prologue(&mut self, frame: &StackFrame, alloc: &RegAllocation) {
        // push rbp → rsp_offset 已被 init_x86_sysv 计算在内
        self.asm.push(rbp);
        self.asm.mov(rbp, rsp);
        // 此时 vm_state.rsp_offset = 0（rbp = rsp）

        // Push callee-saved → 跟踪每个 push 的影响
        for &reg in &alloc.callee_saved_used {
            self.vm_state.rsp_offset -= 8;
            self.vm_state.callee_save_locations.push((reg, self.vm_state.rsp_offset));
            self.asm.push(Self::gpr(reg));
        }

        // sub rsp → 跟踪帧大小
        if frame.total_size > 0 {
            self.vm_state.rsp_offset -= frame.total_size as i32;
            self.asm.sub(rsp, frame.total_size as i32);
        }

        // spill 区域起始 = callee_save 之后
        self.vm_state.spill_base = -(alloc.callee_saved_used.len() as i32 * 8);
    }
}
```

### 3.2 参数加载通过 VM 状态查询

```rust
impl VmState {
    /// 查询参数的物理位置——返回 PtrExpr 供 VmInstr::LoadPtr 使用。
    ///
    /// 禁止在外部硬编码 PtrExpr::StackArg(24) 等——
    /// 必须通过此方法获取（ARCH-VM-QUERY-NOT-ASSUME）。
    pub fn arg_ptr_expr(&self, name: &str) -> Result<PtrExpr, CompilerError> {
        let loc = self.arg_locations.get(name)
            .ok_or_else(|| CompilerError::CodegenViolation(
                format!("Unknown ABI parameter: '{}'", name)
            ))?;
        Ok(match loc {
            ArgLocation::Register(gpr) => PtrExpr::AbiArg(gpr.0),
            ArgLocation::Stack(off) => PtrExpr::StackArg(*off),
        })
    }
}
```

## 4. SymDim 绑定与 VM 状态的统一

### 4.1 SymDimSlotMap 从 VmState 构建

```rust
impl SymDimSlotMap {
    /// 从 VmState 构建——所有位置从 VM 状态查询，零硬编码。
    pub fn from_vm_state(state: &VmState) -> Result<Self, CompilerError> {
        let mut slots = HashMap::new();
        // 固定参数
        for name in ["input", "weights", "kv_cache", "positions",
                      "seq_lens", "output", "scratchpad", "telemetry"] {
            slots.insert(name.into(), state.arg_ptr_expr(name)?);
        }
        // 符号维度
        slots.insert("seq_len".into(), state.arg_ptr_expr("seq_len")?);
        slots.insert("batch_size".into(), state.arg_ptr_expr("batch_size")?);
        slots.insert("total_seq".into(), state.arg_ptr_expr("seq_len")?);
        Ok(Self { slots })
    }
}
```

### 4.2 BoundExpr::Symbolic 解析

ISA Lower 阶段，`BoundExpr::Symbolic` 通过 `SymDimSlotMap::resolve()` → `PtrExpr` → 物理指令。
`SymDimSlotMap` 来自 `VmState`，`VmState` 来自平台 ABI 初始化 + prologue 状态跟踪。

**完整链路（零硬编码）**:
```
平台 ABI 规则 (x86 SysV)
  → VmState::init_x86_sysv(COMPILED_LAYER_PARAM_NAMES)
    → arg_locations: {"output" → Stack(24), "seq_len" → Stack(16), ...}
      （24 = stack_arg_base(16) + (7-6)*8 = 24，计算结果）
  → SymDimSlotMap::from_vm_state(&vm_state)
    → slots: {"output" → StackArg(24), "seq_len" → StackArg(16), ...}
  → plan_lower: sym_map.resolve("output") → StackArg(24)
  → lower: BoundExpr::Symbolic("seq_len") → sym_map.resolve → StackArg(16)
  → x86_lower: cmp counter, [rbp+16]
```

数字 `24` 和 `16` 出现在运行时内存中，但**没有出现在源代码的任何字面量中**——它们是 `stack_arg_base + (idx - 6) * 8` 的计算结果。

## 5. 废弃清单

| 废弃 | 替代 | 理由 |
|------|------|------|
| `executable.rs::abi_slots` 模块 | `VmState::arg_ptr_expr()` | 硬编码常量 → VM 状态动态查询 |
| `AbiSlot::Stack(24)` | `VmState::init_x86_sysv()` 计算 | 硬编码偏移 → ABI 推导 |
| `SymDimSlotMap::from_abi()` | `SymDimSlotMap::from_vm_state()` | 静态构建 → VM 状态驱动 |
| `X86Lower::with_sym_map(avx512, map)` | `X86Lower::with_vm_state(avx512, vm_state)` | Map 传入 → State 传入 |
| 所有 `PtrExpr::StackArg(数字)` 字面量 | `vm_state.arg_ptr_expr("name")` | 硬编码 → 查询 |
| 所有 `PtrExpr::AbiArg(数字)` 字面量 | `vm_state.arg_ptr_expr("name")` | 硬编码 → 查询 |

## 6. 铁律

| 铁律 | 说明 |
|------|------|
| **ARCH-VM-NO-MAGIC-OFFSET** | 源代码中禁止出现 `StackArg(16)` / `StackArg(24)` / `AbiArg(4)` 等数字字面量。所有偏移由 VmState 计算。 |
| **ARCH-VM-PROLOGUE-TRACKS** | prologue 中每条修改 rsp 的指令必须更新 VmState.rsp_offset。 |
| **ARCH-VM-QUERY-NOT-ASSUME** | 参数位置通过 `vm_state.arg_ptr_expr("name")` 查询。查询失败返回 Err，禁止 fallback。 |
| **ARCH-VM-PLATFORM-INIT** | VmState 的初始化由平台 ABI 规则驱动（`init_x86_sysv` / `init_aarch64_aapcs` / `init_gpu_kernel`），不手写。 |

## 7. 实现路径

### Phase 1: VmState 结构 + init_x86_sysv
- 在 `vm/` 模块新建 `vm_state.rs`
- 定义 `VmState`、`ArgLocation`、`COMPILED_LAYER_PARAM_NAMES`
- 实现 `init_x86_sysv()` 和 `arg_ptr_expr()`

### Phase 2: prologue 状态跟踪
- 修改 `X86Lower::emit_prologue()` 接收 `&mut VmState`
- 每条 push/sub 更新 `vm_state.rsp_offset` 和 `callee_save_locations`

### Phase 3: SymDimSlotMap 从 VmState 构建
- 修改 `SymDimSlotMap::from_vm_state(&VmState)` 替代 `from_abi()`
- 所有 `sym_map.resolve()` 调用不变（接口不变，来源变了）

### Phase 4: 删除 abi_slots 硬编码
- 删除 `executable.rs::abi_slots` 模块和 `AbiSlot` 枚举
- 删除所有 `abi_slots::XXX.to_ptr_expr()` 引用
- grep 确认零 `StackArg(数字)` / `AbiArg(数字)` 字面量残留
