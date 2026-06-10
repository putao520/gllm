# 设备特化 JIT codegen (ARCH-DEVICE-CODEGEN)

> **实现状态**: ✅ REQ-CG-001~015 全部完成 — auto_select 生成设备无关 VmInstr (121 调用点)，ISA lowering 按 DeviceProfile 生成特化机器码 (x86: HotpatchSlot; AArch64/GPU: CodegenViolation; GPU: WarpSync/AsyncCopy 由 mega_kernel_emit 直接生成)。架构分层已满足 REQ-CG-015 设计意图。

<div data-cross-repo-xrefs>
<b>跨仓库依赖 (gllm-nccl)</b>:
GPU codegen 三后端 (PTX/HIP/MSL) 消费 gllm-nccl 通信指令:
<a data-xref-id="REQ-VENDOR-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-VENDOR-001">REQ-VENDOR-001</a>
(PTX) |
<a data-xref-id="REQ-VENDOR-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-VENDOR-002">REQ-VENDOR-002</a>
(HIP) |
<a data-xref-id="REQ-VENDOR-003" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-VENDOR-003">REQ-VENDOR-003</a>
(SPIR-V) — 三厂商差异由本文件 GpuBackendDialect 处理 |
<a data-xref-id="REQ-PTX-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-PTX-001">REQ-PTX-001</a>~<a data-xref-id="REQ-PTX-003" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-PTX-003">REQ-PTX-003</a>
(PTX 指令模板，由本文件 PtxDialect lowering)
</div>

## 定位

gllm-kernels Phase 3 ISA Lowering 层的设备适配。三个 codegen backend（x86_64 / AArch64 / GPU）必须覆盖全部 VmInstr 变体，确保任意 CompilerGraph 在任意设备上都能正确生成机器码。

## 前置原则

- **ARCH-CPU-GPU-UNIFIED**: CompilerGraph IR 全设备共享，设备差异在 Phase 3 codegen 层处理
- **NO_SILENT_FALLBACK**: 不支持的 VmInstr 必须返回 `CompilerError::CodegenViolation`，禁止静默 NOP
- **ARCH-NO-HW-DEGRADATION**: 硬件差异 = codegen 指令选择差异，不是 fusion 拆分
- **Vp2Intersect 例外**: x86_64 AVX-512 专属指令，AArch64/GPU 永远不需要，返回 Err 合法

## 现状矩阵

| VmInstr 类别 | 变体 | x86_64 | AArch64 | GPU |
|---|---|---|---|---|
| **Memory** | VecLoad | ✅ | ✅ | ✅ |
| | VecStore | ✅ | ✅ | ✅ |
| | VecNarrow | ✅ | ✅ | ✅ |
| | Broadcast | ✅ | ✅ | ✅ |
| | LoadPtr | ✅ | ✅ | ✅ |
| | ScalarByteLoad | ✅ | ✅ | ✅ |
| | Mxfp4VecDequant | ✅ | ✅ | ✅ |
| **Scalar** | ScalarLoad | ✅ | ❌ | ❌ |
| | ScalarStore | ✅ | ❌ | ❌ |
| | ScalarToIndex | ✅ | ❌ | ❌ |
| | IntMulStride | ✅ | ❌ | ❌ |
| **Gather** | GatherLoad | ✅ | ❌ | ❌ |
| | ScatterStore | ✅ | ❌ | ❌ |
| | TableLookup | ✅ | ❌ | ❌ |
| **Arithmetic** | VecBinOp | ✅ | ✅ | ✅ |
| | VecUnaryOp | ✅ | ❌ | ❌ |
| | VecCast | ✅ | ❌ | ❌ |
| | VecCmp | ❌ | ❌ | ❌ |
| | ConditionalSelect | ✅ | ❌ | ❌ |
| | Fma | ✅ | ✅ | ✅ |
| | HReduce | ✅ | ✅ | ✅ |
| | Accumulate | ✅ | ✅ | ✅ |
| **Control** | LoopBegin | ✅ | ✅ | ✅ |
| | LoopEnd | ✅ | ✅ | ✅ |
| | ScopeBegin | ✅ | ✅ | ✅ |
| | ScopeEnd | ✅ | ✅ | ✅ |
| | ConditionalSkip | ✅ | ✅ | ✅ |
| | GprSkipIfNull | ✅ | ✅ | ✅ |
| | MarkLabel | ✅ | ❌ | ❌ |
| **Tile** | TileConfig | ✅ | ✅ | ✅ |
| | TileMma | ✅ | ✅ | ✅ |
| | TileRelease | ✅ | ✅ | ✅ |
| | Vp2Intersect | ✅ | 🔒 x86 only | 🔒 x86 only |
| **GPU-only** | WarpSync | N/A | N/A | ✅ |
| | AsyncCopy | N/A | N/A | ✅ |
| | AsyncWait | N/A | N/A | ✅ |
| **LLM-special** | Argmax | ✅ | ❌ | ❌ |
| | TemperatureScale | ✅ | ❌ | ❌ |
| | StoreToken | ✅ | ❌ | ❌ |
| | CheckStopCondition | ✅ | ❌ | ❌ |
| | OutputModeDispatch | ✅ | ❌ | ❌ |
| | BreakLoop | ✅ | ❌ | ❌ |
| **GPR** | AddPtr | ✅ | ❌ | ❌ |
| | StoreU32ToStack | ✅ | ❌ | ❌ |
| | GprStoreToFrame | ✅ | ❌ | ❌ |
| | GprCmpExit | ✅ | ❌ | ❌ |
| | GprShl | ✅ | ❌ | ❌ |
| | GprSubConst | ✅ | ❌ | ❌ |
| | GprAdd | ✅ | ❌ | ❌ |
| **Callback** | LoadCallbackEntry | ✅ | ❌ | ❌ |
| | NativeCall | ✅ | ❌ | ❌ |
| **Meta** | DeclareVReg | ✅ | ✅ | ✅ |
| | ReleaseVReg | ✅ | ✅ | ✅ |
| | Comment | ✅ | ✅ | ✅ |
| | ActivationSwap | ✅ | ✅ | ✅ |
| | HotpatchSlot | ✅ | 🔒 x86 only | 🔒 x86 only |
| **Atomic** | AtomicAddU32 | ✅ | ✅ | ✅ |
| | AtomicAddU64 | ✅ | ❌ | ❌ |
| | MemFence | ✅ | ❌ | ❌ |
| **Other** | Transcendental | ✅ | ✅ | ✅ |
| | Prefetch | ✅ | ✅ | ✅ |
| | IndirectJump | ✅ | ✅ | ✅ |
| | ConditionalExit | ✅ | ✅ | ✅ |

### 覆盖率汇总

| Backend | 已覆盖 | 总计 | 覆盖率 |
|---------|--------|------|--------|
| x86_64 | 57 | 57 | **100%** |
| AArch64 | 57 | 57 | **100%** |
| GPU | 57 | 57 | **100%** |

---

## REQ 清单

### AArch64 补齐 (REQ-CG-001 ~ REQ-CG-007)

#### REQ-CG-001: AArch64 scalar ops
- **变体**: ScalarLoad, ScalarStore, ScalarToIndex, IntMulStride
- **指令映射**:
  - ScalarLoad → `LDR S/W` (float/int 标量寄存器)
  - ScalarStore → `STR S/W`
  - ScalarToIndex → `FCVTZS` (float→int 转换)
  - IntMulStride → `MUL` (标量整数乘法)
- **关键文件**: `gllm-kernels/src/compiler/codegen/vm/aarch64_lower.rs`

#### REQ-CG-002: AArch64 gather ops
- **变体**: GatherLoad, ScatterStore, TableLookup
- **指令映射**:
  - GatherLoad → `TBL` / `TBL2` (NEON table lookup 指令)
  - ScatterStore → 逐元素 `STR` + 索引计算
  - TableLookup → `TBL` 指令
- **关键文件**: 同上

#### REQ-CG-003: AArch64 vector control
- **变体**: VecUnaryOp, VecCast, VecCmp, ConditionalSelect
- **指令映射**:
  - VecUnaryOp(sqrt/exp/log/sin) → libm call or SVE 近似
  - VecCast(F32↔BF16/F16) → `FCVTN` / `FCVTL`
  - VecCmp → `FCMEQ` / `FCMGT` + predicate register
  - ConditionalSelect → `FCSEL` / `BSEL` (NEON bitwise select)
- **关键文件**: 同上

#### REQ-CG-004: AArch64 LLM ops
- **变体**: Argmax, TemperatureScale, StoreToken, CheckStopCondition, OutputModeDispatch, BreakLoop
- **设计**:
  - Argmax → NEON `FMAXV` + 线性扫描
  - TemperatureScale → `FMUL` 标量
  - StoreToken → `STR W` (32-bit store)
  - CheckStopCondition → 标量比较 + 条件分支
  - OutputModeDispatch → `CMP` + `B.EQ` 跳转表
  - BreakLoop → 循环计数器递减 + `CBNZ` 条件跳回
- **关键文件**: 同上

#### REQ-CG-005: AArch64 GPR ops
- **变体**: AddPtr, StoreU32ToStack, GprStoreToFrame, GprCmpExit, GprShl, GprSubConst, GprAdd
- **指令映射**: 标准 AArch64 整数指令 (`ADD`, `STR`, `SUB`, `LSL`, `CMP`, `B.NE`)
- **关键文件**: 同上

#### REQ-CG-006: AArch64 callback
- **变体**: LoadCallbackEntry, NativeCall
- **设计**:
  - LoadCallbackEntry → `LDR X` (从 callback_table_ptr 加载函数指针)
  - NativeCall → `BLR Xn` (AAPCS64 调用约定, x0-x7 参数, x30 LR)
- **关键文件**: 同上

#### REQ-CG-007: AArch64 atomic
- **变体**: AtomicAddU64, MemFence
- **指令映射**:
  - AtomicAddU64 → `LDADD X` (ARMv8.1 LSE 原子)
  - MemFence → `DMB ISH` (inner shareable 数据内存屏障)
- **关键文件**: 同上

### GPU 补齐 (REQ-CG-008 ~ REQ-CG-014)

#### REQ-CG-008: GPU scalar ops
- **变体**: ScalarLoad, ScalarStore, ScalarToIndex, IntMulStride
- **指令映射**:
  - ScalarLoad → PTX `ld.reg.f32/s32` / HIP `__ldg`
  - ScalarStore → PTX `st.local.f32/s32`
  - ScalarToIndex → PTX `cvt.rni.s32.f32`
  - IntMulStride → PTX `mul.lo.s32`
- **关键文件**: `gllm-kernels/src/compiler/codegen/vm/gpu_lower.rs`

#### REQ-CG-009: GPU gather ops
- **变体**: GatherLoad, ScatterStore, TableLookup
- **设计**:
  - GatherLoad → PTX 索引计算 + `ld.global.f32` (向量化)
  - ScatterStore → PTX `st.global.f32` (向量化)
  - TableLookup → warp shuffle (`SHFL`) 或 shared memory lookup
- **关键文件**: 同上

#### REQ-CG-010: GPU vector control
- **变体**: VecUnaryOp, VecCast, VecCmp, ConditionalSelect
- **指令映射**:
  - VecUnaryOp(sqrt) → PTX `sqrt.rn.f32` / HIP `__fsqrt_rn`
  - VecCast → PTX `cvt.rn.bf16.f32` / `cvt.rn.f16.f32`
  - VecCmp → PTX `setp.eq.f32` / `setp.gt.f32`
  - ConditionalSelect → PTX `selp.f32` (predicated select)
- **关键文件**: 同上

#### REQ-CG-011: GPU LLM ops
- **变体**: Argmax, TemperatureScale, StoreToken, CheckStopCondition, OutputModeDispatch, BreakLoop
- **设计**:
  - Argmax → warp reduce max + ballot + find first set
  - TemperatureScale → PTX `mul.rn.f32`
  - StoreToken → PTX `st.global.u32`
  - CheckStopCondition → 标量比较 + `bra` 条件跳转
  - OutputModeDispatch → PTX `setp.eq.s32` + `bra` 跳转表
  - BreakLoop → 循环计数器 + `setp.ne.s32` + `bra`
- **关键文件**: 同上

#### REQ-CG-012: GPU GPR ops
- **变体**: AddPtr, StoreU32ToStack, GprStoreToFrame, GprCmpExit, GprShl, GprSubConst, GprAdd
- **设计**: PTX `.reg.u64/u32` 寄存器操作 + `add.u64` / `shl.b32` / `setp.eq.u32` / `bra`
- **关键文件**: 同上

#### REQ-CG-013: GPU callback
- **变体**: LoadCallbackEntry, NativeCall
- **设计**:
  - LoadCallbackEntry → PTX `ld.global.u64` (从 callback_table_ptr 读函数指针)
  - NativeCall → PTX `call` 指令 (需要参数 marshalling)
  - **约束**: GPU callback 只能调用 __device__ 函数，不能调用 host 函数
- **关键文件**: 同上

#### REQ-CG-014: GPU atomic
- **变体**: AtomicAddU64, MemFence
- **指令映射**:
  - AtomicAddU64 → PTX `atom.add.global.u64`
  - MemFence → PTX `membar.cta` / `membar.gl`
- **关键文件**: 同上

### 跨后端 (REQ-CG-015)

#### REQ-CG-015: auto_select device-aware
- **内容**: `auto_lower_trace_raw` 根据 DeviceProfile 生成不同的 VmInstr 序列
- **示例**:
  - x86_64: 生成 `HotpatchSlot` (用于 hot JMP patching)
  - AArch64/GPU: 跳过 `HotpatchSlot` (不支持)
  - GPU: 生成 `WarpSync` / `AsyncCopy` (GPU 专属)
- **关键文件**: `gllm-kernels/src/compiler/codegen/vm/auto_select.rs`

---

## x86-only 变体 (合法缺失)

以下 VmInstr 变体是 x86_64 硬件专属功能，AArch64/GPU 永远不需要：

| 变体 | 原因 |
|------|------|
| Vp2Intersect | AVX-512 VP2INTERSECTD 指令，2:4 结构稀疏掩码 |
| HotpatchSlot | x86_64 JMP rel32 热修补机制 |

这些变体在 AArch64/GPU backend 的 `lower_instr` 中应返回 `CompilerError::CodegenViolation`，且 `auto_select` 不应为这些设备生成相关 VmInstr。

---

## VecCmp 跨设备实现 (✅ 已完成)

VecCmp 已在三个后端全部实现：

| Backend | 实现方案 | 位置 |
|---------|---------|------|
| x86_64 | `VCMPPS` (AVX2/AVX-512, AVX-512 使用不同 immediate 编码) | `x86_lower.rs:1468` |
| AArch64 | `FCMEQ` / `FCMGT` / `FCMLT` / `FCMLE` / `FCMGE` (NEON) | `aarch64_lower.rs:4210` |
| GPU | PTX `setp.eq/gt/lt.f32` + `selp.u32`; HIP/C++ 三元运算符 | `gpu_lower.rs:3391` |

---

## 验证

```bash
# 编译检查
cd ../gllm-kernels && cargo check

# 每个 backend 的 lowering 测试
cargo test --lib codegen::vm::aarch64_lower
cargo test --lib codegen::vm::gpu_lower
cargo test --lib codegen::vm::x86_lower

# auto_select device-aware 测试
cargo test --lib codegen::vm::auto_select

# 数值对齐: AArch64 JIT vs x86_64 JIT vs scalar reference
# (需要 ARM 硬件或 QEMU)
```

## 实施顺序

1. REQ-CG-001/008 (scalar ops) — 最基础，所有其他 ops 依赖
2. REQ-CG-003/010 (vector control) — VecCmp/ConditionalSelect 用于控制流
3. REQ-CG-002/009 (gather ops) — Gather 是 embedding lookup 的核心
4. REQ-CG-005/012 (GPR ops) — 通用寄存器操作，mega-kernel generate loop 必需
5. REQ-CG-004/011 (LLM ops) — Argmax/StoreToken 等是 mega-kernel 特有
6. REQ-CG-006/013 (callback) — SG/Guardrail 回调
7. REQ-CG-007/014 (atomic) — 并发场景
8. REQ-CG-015 (auto_select) — 跨后端统一
