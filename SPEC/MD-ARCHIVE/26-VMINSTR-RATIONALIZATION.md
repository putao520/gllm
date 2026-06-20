# 26-VMINSTR-RATIONALIZATION.md

**VmInstr 去类型化合并 + 缺失指令补全**

> **SSOT**: 本文档是 VmInstr 指令集去类型化合并、缺失指令补全、TraceOp→VmInstr 映射完整性、ISA Lowering 重构的**唯一权威源**。
>
> **关联 SPEC**: 01-JIT-PIPELINE.md (ARCH-AUTO-INSTR-SELECT), 23-QUANT-CODEGEN-ALGO.md, 24-QUANT-PIPELINE-JIT.md, 25-JIT-LIFECYCLE-INFRASTRUCTURE.md

## §0 设计动机

### 问题诊断

VmInstr 枚举当前 142 个变体，存在三类系统性问题：

| 问题类别 | 症状 | 量化影响 |
|----------|------|---------|
| **dtype 硬编码在名称中** | `Bf16Dot`/`Fp16Dot`/`Int8Dot` 是同一条指令的 3 个 dtype 特化 | 每新增一个 dtype = 新增 N 个变体 |
| **功能域重复** | `QuantVecBitAnd` ≡ `VecBinOp::And`, `QuantFma` ≡ `Fma`, `QuantScalarLoad` ≡ `ScalarLoad` | 18 个 Quant* 变体中 10 个与通用指令语义完全重复 |
| **语义缺口** | `TraceOp::Permute` 用 identity copy 绕过（无 VecShuffle）、无 GPR 逻辑运算、无向量 lane 提取/插入 | GPU codegen 和高级融合被迫 work around |

### ARCH-VMINSTR-TYPE-ERASED

**设计原则（参照 Triton TTIR）**：dtype 在类型签名中，不在指令名中。Triton 用 ~20 个 type-erased op（`tt.load`/`tt.store`/`arith.addf`）覆盖所有 dtype，gllm 应同等参数化。

**对照表**：

| 系统 | 指令数 | dtype 策略 | gllm 对标 |
|------|--------|-----------|-----------|
| Triton TTIR | ~20 | 类型签名 (`!tt.ptr<f32>`) | VecLoad { dtype } ✅ 已实现 |
| LLVM SelectionDAG | ~60 | EVT (Value Type) | VecBinOp { op, dtype } ✅ 已实现 |
| gllm VmInstr (当前) | 142 | 混合（部分参数化，部分硬编码） | 需统一 |
| gllm VmInstr (目标) | ~97 | 全参数化 + 补全 | 本 SPEC |

---

## §1 去类型化合并 (REQ-VR-001~004)

### §1.1 GGUF 量化块加载合并 (REQ-VR-001)

**当前**：11 个 GgufXxxLoad 变体，全部是 `load bytes → unpack → dequant → compute_dtype`，仅解包算法不同。compute_dtype 由 TensorMeta.dtype × DeviceProfile 推导，量化路径计算精度从权重 dtype 推导(多精度混合支持 F32/BF16/F16)，原生浮点路径保持原始精度。

**目标**：2 个参数化变体 + 保留 2 个 scale 加载。

#### §1.1.1 QuantBlockLoad — 单平面量化块加载

```
QuantBlockLoad:
  输入:
    dst: VRegId            — 目标 Vec 寄存器 (compute_dtype, 由 TensorMeta.dtype × DeviceProfile 推导)
    base: VRegId           — 基地址 GPR
    offset: OffsetExpr     — 字节偏移
    unpack: BlockUnpackMode — 解包模式（完整描述 byte→i32→compute_dtype 策略）
    width: SimdWidth       — SIMD 宽度
    dtype: QuantPrecision  — 输出计算精度 (从 op_input_dtype 传播, 非 F32 硬编码)

BlockUnpackMode:           — 替代 7 个 GgufXxxLoad 变体
  | Int8                   — i8 → compute_dtype 直转 (原 GgufInt8Load)
  | F16Broadcast           — f16 → compute_dtype + broadcast (原 GgufF16ScaleLoad)
  | SignedNibbleLow        — & 0x0F, sub 8.0 (原 GgufInt4Load)
  | UnsignedNibbleLow      — & 0x0F, no bias (原 GgufUInt4Load)
  | SignedNibbleHigh       — >> 4, sub 8.0 (原 GgufInt4HighLoad)
  | UnsignedNibbleHigh     — >> 4, no bias (原 GgufUInt4HighLoad)
  | Bitpack2               — 2-bit ×4 per byte, sub bias (原 GgufInt2Load)
```

**删除的变体**：GgufInt8Load, GgufF16ScaleLoad, GgufInt4Load, GgufUInt4Load, GgufInt4HighLoad, GgufUInt4HighLoad, GgufInt2Load

**x86_lower 分发**：按 `unpack` 值 match，内部逻辑不变（只是从 7 个 VmInstr match arm 变为 1 个 + 内部 match on BlockUnpackMode）。

#### §1.1.2 QuantBiPlaneLoad — 多平面位合并加载

```
QuantBiPlaneLoad:          — 替代 3 个多平面 GgufXxxLoad 变体
  输入:
    dst: VRegId            — 目标 Vec 寄存器 (compute_dtype, 同 QuantBlockLoad)
    qs_base: VRegId        — 低 bit 平面基地址
    extra_base: VRegId     — 高 bit 平面基地址 (qh / hmask)
    bias: f32              — 解码后减去的偏置 (恒为 f32, 量化偏置参数)
    mode: BiPlaneMode      — 合并策略
    width: SimdWidth
    dtype: QuantPrecision  — 输出计算精度 (同 QuantBlockLoad.dtype)

BiPlaneMode:
  | Low5                   — qs(4-bit) | (qh(1-bit) << 4) (原 GgufInt5Load)
  | Low6                   — qs(4-bit) | (qh(2-bit) << 4) (原 GgufInt6Load)
  | Q3Merge                — qs(2-bit) | (hmask(1-bit) << 2) (原 GgufInt3Load)
```

**删除的变体**：GgufInt5Load, GgufInt6Load, GgufInt3Load

#### §1.1.3 保留不变的 Scale 加载

```
GgufSubScaleLoad           — 保留，语义独特（sub-block i8 scale 广播）
GgufKQuantScaleLoad        — 保留，语义独特（K-Quant 6-bit packed scale 解码）
```

**变更汇总**：11 → 4（-7）

---

### §1.2 Dot-Product 合并 (REQ-VR-002)

**当前**：5 个 DotProduct 变体（Bf16Dot/Fp16Dot/Int8Dot/Int4x8Dot/Fp4Dot），差异仅在输入 dtype。

**目标**：1 个参数化变体。

```
DotProduct:                — 替代 5 个 dtype-specific Dot 变体
  输入:
    acc: VRegId            — 累加器 (in/out)
    a: VRegId              — 输入 A
    b: VRegId              — 输入 B
    input_dtype: DotDtype  — 输入数据类型
    width: SimdWidth

DotDtype:
  | Bf16                   — fp32 acc += bf16·bf16 (原 Bf16Dot)
  | Fp16                   — fp32 acc += fp16·fp16 (原 Fp16Dot)
  | Int8                   — int32 acc += int8·int8 (原 Int8Dot)
  | Int4x8                 — int32 acc += int4·int8 (原 Int4x8Dot)
  | Fp4                    — fp32 acc += e2m1·e2m1 (原 Fp4Dot)
```

**ScaleApplyInt** 保留不变 — 它是累加器缩放（不是 dot-product）。

**变更汇总**：5 → 1（+ 保留 ScaleApplyInt = 2，-3）

---

### §1.3 GPR 整数操作合并 (REQ-VR-003)

**当前**：11 个 Gpr* 变体，大部分是标量整数算术。

**目标**：3 个参数化变体。

#### §1.3.1 GprBinOp — 通用 GPR 二元操作

```
GprBinOp:                  — 替代 GprAdd/GprSub/GprMulConst/GprSubConst/GprShl/GprShr/GprBitTest
  输入:
    dst: VRegId
    a: VRegId
    b: GprOperand          — VReg 或 立即数
    op: GprOp              — 操作类型

GprOperand:
  | VReg(VRegId)
  | Imm(i64)

GprOp:
  | Add                    — dst = a + b
  | Sub                    — dst = a - b
  | Mul                    — dst = a * b
  | Div                    — dst = a / b (截断)
  | Shl                    — dst = a << b
  | Shr                    — dst = a >> b (算术右移)
  | And                    — dst = a & b
  | Or                     — dst = a | b
  | Xor                    — dst = a ^ b
  | BitTest                — dst = (a >> b) & 1 (零标志位设置)
```

**删除的变体**：GprAdd, GprSub, GprMulConst, GprSubConst, GprShl, GprShr, GprBitTest, QuantIntDivConst, QuantIntMul

**新增能力**：GprAnd/GprOr/Xor（当前缺失，GPR 无法做位与/位或/异或）、Div（当前仅 QuantIntDivConst 有除法）。

#### §1.3.2 GprLoadImm — 保留不变

```
GprLoadImm { dst, value: u64 }  — 无二元形式，保留
```

#### §1.3.3 GprCondAction — GPR 条件操作合并

```
GprCondAction:             — 替代 GprSkipIfNull + GprCmpExit + GprStoreToFrame
  输入:
    cond: GprCondition     — 条件判断
    action: GprBranchAction — 条件为真时的行为

GprCondition:
  | IsNull(gpr)            — gpr == 0
  | IsNonNull(gpr)         — gpr != 0
  | CmpLt(a, b)            — a < b
  | CmpEq(a, imm)          — a == imm
  | CmpGte(a, b)           — a >= b

GprBranchAction:
  | Skip(usize)            — 跳过后续 N 条指令
  | Exit(LabelId)          — 跳转到 label
  | StoreFrame { src, rbp_offset }  — 存储到栈帧
```

**变更汇总**：11 → 3（+ 吸收 QuantIntDivConst/QuantIntMul = -9 净减）

---

### §1.4 Quant* 解码指令去重 (REQ-VR-004)

**当前**：18 个 Quant* 变体。其中 10 个与通用指令语义完全重复。

#### §1.4.1 删除 — 复用已有通用指令

| 删除的 Quant* 变体 | 复用目标 | 映射 |
|---|---|---|
| QuantVecBitAnd | VecBinOp { op: And, dtype: I32 } | 位操作 = 整数二元操作 |
| QuantVecBitOr | VecBinOp { op: Or, dtype: I32 } | 同上 |
| QuantBroadcast | Broadcast { dtype: F32 } | 完全相同语义 |
| QuantFma | Fma | 完全相同语义 |
| QuantScalarLoad | ScalarLoad | 完全相同语义 |
| QuantIntDivConst | GprBinOp { op: Div, b: Imm(divisor) } | 整数除法 |
| QuantIntMul | GprBinOp { op: Mul, b: Imm(factor) } | 整数乘法 |
| QuantVecShiftLeft | VecBinOp { op: Shl, dtype: I32 } | 整数向量左移 |
| QuantVecShiftRight | VecBinOp { op: Shr, dtype: I32 } | 整数向量右移 |
| QuantLoadF16toF32 | QuantLoadI8toF32 扩展为 QuantScalarCvtLoad | 见 §1.4.2 |

#### §1.4.2 合并 — QuantLoadF16toF32 + QuantLoadI8toF32 → QuantScalarCvtLoad

```
QuantScalarCvtLoad:        — 替代 QuantLoadF16toF32 + QuantLoadI8toF32
  输入:
    dst: VRegId
    base: VRegId
    offset: i64
    src_dtype: ScalarCvtSource  — 源数据类型
    width: SimdWidth

ScalarCvtSource:
  | F16                    — f16 → f32 + broadcast
  | I8                     — i8 → f32 + broadcast
  | U8                     — u8 → f32 + broadcast (新增)
```

#### §1.4.3 保留 — 语义独特

```
QuantBroadcastInt          — 整数常量广播，无 ScalarExpr 对应
QuantLoadBytesVec          — 多字节零扩展向量加载，无通用对应
QuantCodebookLookup        — SqueezeLLM 查表，独特语义
QuantExtractBits           — 位域提取，独特语义
QuantDequantFma            — 融合反量化+FMA，核心优化指令
QuantInterleave            — nibble 交叉合并，无通用对应
```

**变更汇总**：18 → 7（-11）

---

### §1.5 VecBinOp 扩展 (REQ-VR-004 续)

为吸收 QuantVecShiftLeft/Right 和 GPR 位操作，扩展 VecOp：

```
VecOp (扩展):
  | Add | Sub | Mul | Div | Max | Min   — 浮点（已有）
  | And | Or | Xor | AndNot              — 整数位操作（已有）
  | Shl                                  — 整数左移（新增，替代 QuantVecShiftLeft）
  | Shr                                  — 整数右移（新增，替代 QuantVecShiftRight）
  | Not                                  — 整数取反（新增，dst = ~src，一元但用 VecBinOp dst=src 模式）
```

**注意**：Shl/Shr 的 `b` 参数是立即数（shift amount），通过 VecBinOp 的 `b` VReg 传入 — auto_select 在 TraceOp::QuantShiftLeft/Right 映射时用 QuantBroadcastInt 先广播 shift amount 到向量。

---

## §1.6 分布式通信 VmInstr 扩展 (REQ-VR-014, feature = "nccl")

> **关联 SPEC**: gllm-nccl SPEC/02-ARCHITECTURE.md §5.3, gllm-nccl SPEC/07-PTX-INTRINSICS.md
> **feature gate**: `#[cfg(feature = "nccl")]` — 未启用时这些变体不存在于枚举中
> **设计原则**: JIT codegen 生成调用桩（call stub），链接到 gllm-nccl 提供的运行时函数。不内联通信实现（通信是跨设备操作，不能 JIT 内联）。

<div data-cross-repo-xrefs>
<b>跨仓库依赖 (gllm-nccl)</b>:
<a data-xref-id="REQ-PTX-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-PTX-001">REQ-PTX-001</a>
<a data-xref-id="REQ-PTX-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-PTX-002">REQ-PTX-002</a>
<a data-xref-id="REQ-PTX-003" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-PTX-003">REQ-PTX-003</a>
(NVLink/Warp/Barrier PTX 模板) |
<a data-xref-id="REQ-ALG-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-ALG-001">REQ-ALG-001</a>
<a data-xref-id="REQ-ALG-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-ALG-002">REQ-ALG-002</a>
<a data-xref-id="REQ-ALG-003" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-ALG-003">REQ-ALG-003</a>
<a data-xref-id="REQ-ALG-004" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-ALG-004">REQ-ALG-004</a>
(通信算法) |
<a data-xref-id="REQ-VENDOR-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-VENDOR-001">REQ-VENDOR-001</a>
<a data-xref-id="REQ-VENDOR-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-VENDOR-002">REQ-VENDOR-002</a>
<a data-xref-id="REQ-VENDOR-003" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-VENDOR-003">REQ-VENDOR-003</a>
(三厂商后端) |
<a data-xref-id="REQ-DP-010" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-010">REQ-DP-010</a>
(VmInstr 扩展消费者) |
<a data-xref-id="REQ-DP-011" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-011">REQ-DP-011</a>
(CommInstr 扩展)
</div>

### §1.6.1 AllReduceChunk — chunk 级 AllReduce

```
AllReduceChunk:
  输入:
    sendbuf: VRegId       — GPU HBM 发送缓冲区指针
    recvbuf: VRegId       — GPU HBM 接收缓冲区指针
    count: VRegId         — 元素数量
    dtype: CommDType      — 数据类型
    op: ReduceOp          — 归约操作
    rank: VRegId          — 当前 rank ID
    world_size: VRegId    — 总 rank 数
    chunk_idx: VRegId     — chunk 索引（用于 Ring 算法定位）

CommDType:
  | Fp32
  | Fp16
  | Bf16
  | Fp8
  | Int8

ReduceOp:
  | Sum
  | Max
  | Min
  | Product
```

**用途**: Pipeline AllReduce 中 chunk 级归约，实现计算/通信重叠。

**x86_lower/aarch64_lower**: 生成 `call [gllm_nccl_all_reduce_chunk_stub]` 桩调用，参数通过系统 ABI 寄存器传递。

**gpu_lower**: 生成 PTX `call {all_reduce_chunk, ...}` 或 HIP 等价调用。跨设备操作不能内联进单 SM 的 kernel。

### §1.6.2 CommBarrier — 通信同步屏障

```
CommBarrier:
  输入:
    barrier_id: u8        — 屏障标识（PTX named barrier: 0-15）
    thread_count: VRegId  — 参与同步的线程数

  GPU 语义:
    PTX: bar.sync barrier_id, thread_count
    HIP: __syncthreads() 或 warp-aggregated barrier

  CPU 语义:
    pthread_barrier_wait() 或 std::sync::Barrier FFI
```

**用途**: 通信 thread block 与计算 thread block 之间的同步（见 gllm-nccl SPEC/07 §4.4）。

### §1.6.3 NvlinkAsyncCopy — NVLink 异步块拷贝

```
NvlinkAsyncCopy:
  输入:
    dst: VRegId           — 目标地址（GPU HBM，本端或远端 rank 映射地址）
    src: VRegId           — 源地址（GPU HBM）
    len: VRegId           — 拷贝字节数（必须 16-byte 对齐，16 的倍数）
    lane: u8              — NVLink lane（0-17，H100 最多 18 lanes）

  GPU 语义 (PTX):
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
      [dst], [src], len, [barrier_addr]

  CPU 语义:
    无意义 — CPU 端为 NOP（通信由 GPU 驱动执行）
```

**约束**:
- `dst`/`src` 必须 16-byte 对齐
- `len` 必须是 16 的倍数
- 需要 `topology.is_nvlink_fully_connected() == true`（gllm-nccl 运行时校验）

### §1.6.4 变更汇总

| 新增变体 | 数量 | feature gate |
|---------|------|-------------|
| AllReduceChunk | +1 | nccl |
| CommBarrier | +1 | nccl |
| NvlinkAsyncCopy | +1 | nccl |
| CommDType 枚举 (5 variants) | +1 | nccl |
| ReduceOp 枚举 (4 variants) | +1 | nccl |
| **通信域小计** | **+5** | nccl |

**总 VmInstr 变体**: ~97 (基础) + 5 (nccl) = ~102

### §1.6.5 TraceOp→VmInstr 映射 (nccl)

| TraceOp | VmInstr | 条件 |
|---------|---------|------|
| CommAllReduceChunk | AllReduceChunk { ... } | cfg(nccl) |
| CommBarrier | CommBarrier { ... } | cfg(nccl) |
| CommNvlinkCopy | NvlinkAsyncCopy { ... } | cfg(nccl) |

未启用 nccl feature 时，`auto_select.rs` 中这三个 TraceOp match arm 由 `#[cfg(feature = "nccl")]` 门控，编译时不存在。

### §1.7 SM 分区同步 VmInstr (SPEC 32 联动)

SPEC 32 §2 SM 分区 Ping-Pong 使用硬件同步原语实现 decode∥prefill 并行。这些原语**不在 VmInstr 层定义**，而是在 GPU ISA lowering 层直接 emit。

**设计决策**: SM 分区同步（cluster.sync / mbarrier / grid_sync）是物理 CTA 级操作，不经过 VmInstr 虚拟机。原因：
1. 同步粒度是 CTA/warp 级，不是指令级——VmInstr 是指令级抽象
2. 同步原语高度硬件特化（SM90 cluster.sync vs SM80 grid_sync vs SM70 ring barrier），无法用统一 VmInstr 表达
3. SPEC 32 的 SM 分区在 `mega_kernel_emit.rs` 的 CTA 编排层实现，不在算子 lowering 层

**与现有 VmInstr 的关系**:
- `CommBarrier`（§1.6）是分布式通信同步（跨节点），保留为 VmInstr
- SM 分区同步是单 GPU 内同步，在 JIT codegen 的 CTA 编排模板中直接 emit PTX/HIP 指令
- CPU 路径使用 `std::sync::Barrier`（Rust std），不需要 VmInstr

> **注**: 如果未来需要 JIT 内部更灵活的同步控制（如动态 barrier），可以扩展 VmInstr。当前设计中 SM 分区同步是编译时确定的 CTA 编排，不需要运行时 VmInstr。

---

## §2 缺失指令补全 (REQ-VR-005~010)

### §2.1 向量 Shuffle/Permute (REQ-VR-005)

**问题**：`TraceOp::Permute` 当前用 identity copy 绕过，无法生成 `vpshufb`/`vpermd`/`TBL`。

```
VecShuffle:
  输入:
    dst: VRegId
    src: VRegId
    mask: VecShuffleMask    — lane 重排描述
    width: SimdWidth

VecShuffleMask:
  | Const(Vec<u8>)          — 编译时已知 shuffle 掩码
  | Dynamic { ctrl: VRegId } — 运行时掩码寄存器
```

**x86_lower**：Const → `vpshufb` (AVX2) / `vpermq` (AVX-512)；Dynamic → `vpermd`
**aarch64_lower**：Const → `TBL`；Dynamic → `TBL`
**gpu_lower**：`PRMT` (PTX) / `ds_permute` (AMD)

### §2.2 向量 Lane 提取/插入 (REQ-VR-006)

**问题**：当前无法从 Vec 寄存器提取单个 lane 到 Scalar，也无法将 Scalar 插入 Vec 的指定 lane。这对于 Top-K 采样、batch argmax 等场景必需。

```
VecExtractLane:
  输入:
    dst: VRegId            — 目标 Scalar 寄存器
    src: VRegId            — 源 Vec 寄存器
    lane: u8               — lane 索引
    dtype: QuantPrecision  — 元素数据类型

VecInsertLane:
  输入:
    dst: VRegId            — 目标 Vec 寄存器
    src_vec: VRegId        — 源 Vec 寄存器
    src_scalar: VRegId     — 源 Scalar 寄存器
    lane: u8               — 目标 lane 索引
    dtype: QuantPrecision
```

**x86_lower**：VecExtractLane → `vextractf128` + `vmovss`；VecInsertLane → `vinsertf128` + `vmovss`
**aarch64_lower**：VecExtractLane → `INS` / `DUP`；VecInsertLane → `INS`

### §2.3 GPR 位逻辑运算 (REQ-VR-007)

**问题**：当前 GPR 无法做 AND/OR/XOR/NOT。权重加载中的位操作（如 K-Quant scale 解码）被迫绕道 Quant* 变体。

**已在 §1.3 GprBinOp 中覆盖**：GprOp 新增 And/Or/Xor 三种操作。

### §2.4 向量常量加载 (REQ-VR-008)

**问题**：当前加载向量常量需要 Broadcast(ScalarExpr::Const) + 可能的 VecBinOp。对于整数常量掩码（如 0x0F broadcast），需要 QuantBroadcastInt。缺少通用的向量常量加载。

```
VecLoadConst:
  输入:
    dst: VRegId
    values: Vec<u32>       — 每个 lane 的常量值（编译时已知）
    dtype: QuantPrecision  — 值的解释类型
    width: SimdWidth
```

**x86_lower**：values.len() == lanes 且全部相同 → `vbroadcastss`；否则 → `vmovdqu` from stack slot（prologue 预存）
**aarch64_lower**：`DUP` 或 `LD1` from literal pool

### §2.5 原子 CAS (REQ-VR-009)

**问题**：当前只有 AtomicAddU32/AtomicAddU64。缺少 Compare-And-Swap，这对于无锁数据结构（batch 并发、KV cache 页表更新）必需。

```
AtomicCAS:
  输入:
    dst: VRegId            — 旧值 (成功=expected，失败=当前内存值)
    ptr: VRegId            — 目标地址 GPR
    expected: VRegId       — 期望值 GPR
    desired: VRegId        — 新值 GPR
    elem_width: usize      — 4 (u32) 或 8 (u64)
    success_order: MemOrdering
    failure_order: MemOrdering

MemOrdering:
  | Relaxed
  | Acquire
  | Release
  | AcqRel
  | SeqCst
```

**x86_lower**：`lock cmpxchg`（4/8 字节）
**aarch64_lower**：`LDAXR` + `STLXR` 循环
**gpu_lower**：`atom.cas` (PTX)

### §2.6 Popcount / CLZ / ByteSwap (REQ-VR-010)

**问题**：当前只有 BitwiseGemm（内部用 popcnt）但无独立的位计数指令。CLZ 对于快速 log2 计算、ByteSwap 对于端序转换必需。

```
GprUnaryOp:
  输入:
    dst: VRegId
    src: VRegId
    op: GprUnaryOpKind

GprUnaryOpKind:
  | Not                    — dst = ~src (位取反)
  | Popcount               — dst = popcount(src) (人口计数)
  | Clz                    — dst = count_leading_zeros(src)
  | Bswap                  — dst = byte_swap(src) (端序转换)
  | Neg                    — dst = -src (算术取反)
```

**x86_lower**：Not → `not`；Popcount → `popcnt` (SSE4.2+)；Clz → `lzcnt` (ABM)；Bswap → `bswap`
**aarch64_lower**：Not → `MVN`；Popcount → `CNT` + `ADDV`；Clz → `CLZ`；Bswap → `REV`

### §2.7 Mxfp4/Nvfp4 合并到 QuantBlockLoad (REQ-VR-001 扩展)

**问题**：Mxfp4VecDequant 和 Nvfp4SubBlockDequant 是 E2M1 4-bit 反量化的两个变体（差异仅缩放格式：E8M0 vs UE4M3），应合并到 QuantBlockLoad 的 BlockUnpackMode。

```
BlockUnpackMode (扩展):
  | ...（§1.1.1 已列出的 7 种）
  | Mxfp4                  — E2M1 + E8M0 scale (原 Mxfp4VecDequant)
  | Nvfp4                  — E2M1 + UE4M3 scale (原 Nvfp4SubBlockDequant)
```

**注意**：Mxfp4/Nvfp4 需要 scale_byte_src 参数。BlockUnpackMode 为这两种模式时，QuantBlockLoad 需要额外的 scale 寄存器。方案：在 QuantBlockLoad 中新增可选字段 `scale_src: Option<VRegId>`（Mxfp4/Nvfp4 为 Some，其他为 None）。

---

## §3 变更总览

### §3.1 删除的变体（-41）

| REQ | 删除的变体 | 替代方案 |
|-----|-----------|---------|
| VR-001 | GgufInt8Load, GgufF16ScaleLoad, GgufInt4Load, GgufUInt4Load, GgufInt4HighLoad, GgufUInt4HighLoad, GgufInt2Load | QuantBlockLoad |
| VR-001 | GgufInt5Load, GgufInt6Load, GgufInt3Load | QuantBiPlaneLoad |
| VR-001 | Mxfp4VecDequant, Nvfp4SubBlockDequant | QuantBlockLoad (Mxfp4/Nvfp4 mode) |
| VR-002 | Bf16Dot, Fp16Dot, Int8Dot, Int4x8Dot, Fp4Dot | DotProduct |
| VR-003 | GprAdd, GprSub, GprMulConst, GprSubConst, GprShl, GprShr, GprBitTest | GprBinOp |
| VR-003 | GprSkipIfNull, GprCmpExit, GprStoreToFrame | GprCondAction |
| VR-004 | QuantVecBitAnd, QuantVecBitOr, QuantBroadcast, QuantFma, QuantScalarLoad | 复用 VecBinOp/Broadcast/Fma/ScalarLoad |
| VR-004 | QuantIntDivConst, QuantIntMul, QuantVecShiftLeft, QuantVecShiftRight | GprBinOp / VecBinOp |
| VR-004 | QuantLoadF16toF32, QuantLoadI8toF32 | QuantScalarCvtLoad |

### §3.2 新增的变体（+18，含 nccl feature +5）

| REQ | 新增变体 | 说明 |
|-----|---------|------|
| VR-001 | QuantBlockLoad { ..., unpack, scale_src } | 参数化量化块加载 |
| VR-001 | QuantBiPlaneLoad { ..., mode } | 多平面位合并加载 |
| VR-001 | BlockUnpackMode 枚举 (9 variants) | 解包策略描述符 |
| VR-001 | BiPlaneMode 枚举 (3 variants) | 双平面合并策略 |
| VR-002 | DotProduct { ..., input_dtype } | 去类型化点积 |
| VR-002 | DotDtype 枚举 (5 variants) | 点积输入类型 |
| VR-003 | GprBinOp { ..., op, b } | 参数化 GPR 二元操作 |
| VR-003 | GprOperand 枚举 | VReg/立即数 |
| VR-003 | GprOp 枚举 (10 variants) | GPR 操作类型 |
| VR-003 | GprCondAction { cond, action } | 参数化 GPR 条件操作 |
| VR-003 | GprCondition 枚举 / GprBranchAction 枚举 | 条件/行为描述 |
| VR-004 | QuantScalarCvtLoad { ..., src_dtype } | 合并 F16/I8 标量转换加载 |
| VR-004 | ScalarCvtSource 枚举 (3 variants) | 标量转换源类型 |
| VR-005 | VecShuffle { ..., mask } | 向量 lane 重排 |
| VR-006 | VecExtractLane { ..., lane } | 向量 lane 提取 |
| VR-006 | VecInsertLane { ..., lane } | 向量 lane 插入 |
| VR-008 | VecLoadConst { ..., values } | 向量常量加载 |
| VR-009 | AtomicCAS { ... } | 原子比较并交换 |
| VR-010 | GprUnaryOp { ..., op } | GPR 一元操作 |
| VR-010 | GprUnaryOpKind 枚举 (5 variants) | GPR 一元操作类型 |
| VR-014 | AllReduceChunk { ..., dtype, op } | chunk 级 AllReduce（nccl feature） |
| VR-014 | CommBarrier { barrier_id, thread_count } | 通信/计算同步屏障（nccl feature） |
| VR-014 | NvlinkAsyncCopy { dst, src, len, lane } | NVLink 异步块拷贝（nccl feature） |
| VR-014 | CommDType 枚举 (5 variants) | 通信数据类型（nccl feature） |
| VR-014 | ReduceOp 枚举 (4 variants) | 通信归约操作（nccl feature） |

### §3.3 VecOp 枚举扩展

```
VecOp (扩展后):
  Add | Sub | Mul | Div | Max | Min    — 浮点（已有）
  And | Or | Xor | AndNot              — 整数位操作（已有）
  Shl | Shr                            — 整数移位（新增）
```

### §3.4 数量汇总

| 域 | 当前 | 删除 | 新增 | 合并后 | 变化 |
|---|---|---|---|---|---|
| 通用内存 | 11 | 0 | 0 | 11 | 0 |
| GGUF 加载 | 11 | -11 | +2 (QuantBlockLoad + QuantBiPlaneLoad) | 2 (+2 scale 保留 = 4) | -7 |
| Mxfp4/Nvfp4 | 2 | -2 | 0 (合并到 QuantBlockLoad) | 0 | -2 |
| Dot-Product | 5 | -5 | +1 (DotProduct) | 1 (+1 ScaleApplyInt = 2) | -3 |
| GPR 操作 | 11 | -11 | +3 (GprBinOp + GprLoadImm + GprCondAction) | 3 | -8 |
| Quant* 解码 | 18 | -11 | +1 (QuantScalarCvtLoad) | 8 | -10 |
| 通用算术 | 5 | 0 | 0 | 5 | 0 |
| 控制流 | 14 | 0 | 0 | 14 | 0 |
| GPU/Shared | 10 | 0 | 0 | 10 | 0 |
| 采样 | 9 | 0 | 0 | 9 | 0 |
| Batch | 3 | 0 | 0 | 3 | 0 |
| 杂项 | 19 | 0 | 0 | 19 | 0 |
| 调试 | 4 | 0 | 0 | 4 | 0 |
| **缺失补全** | 0 | 0 | +6 (VecShuffle/VecExtractLane/VecInsertLane/VecLoadConst/AtomicCAS/GprUnaryOp) | 6 | +6 |
| **通信 (nccl)** | 0 | 0 | +5 (AllReduceChunk/CommBarrier/NvlinkAsyncCopy/CommDType/ReduceOp) | 5 | +5 |
| **总计** | **142** | **-41** | **+18** | **~102** | **-39 (-27%)** |

---

## §4 TraceOp→VmInstr 映射更新 (REQ-VR-011)

auto_select.rs 中所有被删除 VmInstr 的映射必须更新为新变体。

### §4.1 映射变更表

| TraceOp | 当前 VmInstr | 新 VmInstr |
|---------|-------------|-----------|
| QuantBitAnd | QuantVecBitAnd | VecBinOp { op: And, dtype: I32 } |
| QuantBitOr | QuantVecBitOr | VecBinOp { op: Or, dtype: I32 } |
| QuantBroadcast | QuantBroadcast | Broadcast |
| QuantFma | QuantFma | Fma |
| QuantScalarLoad | QuantScalarLoad | ScalarLoad |
| QuantIntDivConst | QuantIntDivConst | GprBinOp { op: Div, b: Imm(divisor) } |
| QuantIntMul | QuantIntMul | GprBinOp { op: Mul, b: Imm(factor) } |
| QuantShiftLeft | QuantVecShiftLeft | VecBinOp { op: Shl, dtype: I32 } |
| QuantShiftRight | QuantVecShiftRight | VecBinOp { op: Shr, dtype: I32 } |
| QuantLoadF16toF32 | QuantLoadF16toF32 | QuantScalarCvtLoad { src_dtype: F16 } |
| QuantLoadI8toF32 | QuantLoadI8toF32 | QuantScalarCvtLoad { src_dtype: I8 } |
| QuantE2m1LutDecode (nvfp4=false) | Mxfp4VecDequant | QuantBlockLoad { unpack: Mxfp4 } |
| QuantE2m1LutDecode (nvfp4=true) | Nvfp4SubBlockDequant | QuantBlockLoad { unpack: Nvfp4 } |
| Permute | identity copy | VecShuffle |
| QuantDataLoad (SignedPackedInt4/Low) | GgufInt4Load | QuantBlockLoad { unpack: SignedNibbleLow } |
| QuantDataLoad (SignedPackedInt4/High) | GgufInt4HighLoad | QuantBlockLoad { unpack: SignedNibbleHigh } |
| QuantDataLoad (PackedInt4/Low) | GgufUInt4Load | QuantBlockLoad { unpack: UnsignedNibbleLow } |
| QuantDataLoad (PackedInt4/High) | GgufUInt4HighLoad | QuantBlockLoad { unpack: UnsignedNibbleHigh } |
| QuantDataLoad (Int8) | GgufInt8Load | QuantBlockLoad { unpack: Int8 } |
| QuantDataLoad (Bitpack2) | GgufInt2Load | QuantBlockLoad { unpack: Bitpack2 } |
| QuantScaleLoad | GgufF16ScaleLoad | QuantBlockLoad { unpack: F16Broadcast } |
| QuantHighBitsLoad (Low5) | GgufInt5Load | QuantBiPlaneLoad { mode: Low5 } |
| QuantHighBitsLoad (Low6) | GgufInt6Load | QuantBiPlaneLoad { mode: Low6 } |
| QuantHighBitsLoad (Q3Merge) | GgufInt3Load | QuantBiPlaneLoad { mode: Q3Merge } |

### §4.1.1 MLA TraceOp→VmInstr 映射 (SPEC 33 联动)

| TraceOp | VmInstr 组合 | 说明 |
|---------|-------------|------|
| MlaKvCompress | VecMatMul { dtype } + VecStore | hidden_state × W_DKV^T → c_KV (d_c 维) |
| MlaQAbsorb | VecMatMul { dtype } + VecStore | q × W_UK^T → q_absorbed (n_h × d_c 维) |
| MlaVRestore | VecMatMul { dtype } + VecStore | score × UV_W → v_restored (n_h × d 维) |
| MlaAttnScore | VecDotProd { dtype } + VecBinOp { ScaleAdd } | q_absorbed · c_KV → attention scores |
| MlaRopeMerge | VecExtractLane + VecInsertLane | 替换 c_KV 后 d_rope 维为 k_pe |
| MlaAttention | MlaAttnScore + Softmax + MlaVRestore + VecMatMul | 完整 MLA attention 流水线 |

> **注**: MLA VmInstr 复用已有 VecMatMul/VecDotProd/VecBinOp/Softmax 等通用指令，不引入 MLA 专用 VmInstr。MLA 的特殊性体现在 TraceOp 语义（MlaAttnScore 定义了 absorbed attention 的计算步骤）和 CompilerGraph 构建阶段（权重布局、维度推导），不在 VmInstr 层。

### §4.1.2 MTP TraceOp→VmInstr 映射 (SPEC 34 联动)

| TraceOp | VmInstr 组合 | 说明 |
|---------|-------------|------|
| MtpDraft | GprBinOp { Add } + QuantBlockLoad { F16Broadcast } + VecMatMul + Argmax | per-depth: hidden + weight_offset → GEMV → argmax |

> **注**: MTP 复用 VecMatMul (GEMV) + Argmax 通用指令。MTP 的特殊性在 MegaKernelBusinessConfig.mtp_config 控制的 MTP 融合组（per-depth 投影 + argmax 循环），不在 VmInstr 层。

### §4.2 plan_lower.rs 中的 GgufXxxLoad 直接调用

plan_lower.rs 中任何直接 `prog.emit(VmInstr::GgufXxxLoad {...})` 的调用都必须替换为新的 QuantBlockLoad/QuantBiPlaneLoad。通过 grep 审计确认。

---

## §5 ISA Lowering 重构 (REQ-VR-012)

### §5.1 x86_lower.rs

- 删除 7 个 GgufXxxLoad match arm → 替换为 1 个 QuantBlockLoad arm（内部 match on BlockUnpackMode）+ 1 个 QuantBiPlaneLoad arm
- 删除 5 个 DotProduct match arm → 替换为 1 个 DotProduct arm（内部 match on DotDtype）
- 删除 7 个 Gpr* match arm → 替换为 GprBinOp + GprCondAction 两个 arm
- 删除 10 个 Quant* 重复 arm → 复用 VecBinOp/Broadcast/Fma/ScalarLoad arm
- 新增 VecShuffle/VecExtractLane/VecInsertLane/VecLoadConst/AtomicCAS/GprUnaryOp arm

### §5.2 aarch64_lower.rs

- 同 x86_lower.rs 结构，ISA 映射替换为 NEON/SVE 指令

### §5.3 gpu_lower.rs

- 同 x86_lower.rs 结构，ISA 映射替换为 PTX/HIP/MSL 指令

---

## §6 验证策略 (REQ-VR-013)

### §6.1 编译验证

```
cd ../gllm-kernels && cargo check
cd ../gllm && cargo check
```

### §6.2 auto_select 映射完整性

每个 TraceOp 变体必须有且仅有一个 VmInstr 映射。通过 `dispatch_trace_op` 的 exhaustive match 保证。

### §6.3 数值对齐

Q4_0/Q8_0 E2E 测试输出与重构前对比，误差 ≤ 1e-6。

### §6.4 数量审计

```
# 确认删除的变体不再存在
grep -c "GgufInt8Load\|GgufF16ScaleLoad\|GgufInt4Load\|..." instr.rs  → 0

# 确认新增变体存在
grep -c "QuantBlockLoad\|DotProduct\|GprBinOp\|..." instr.rs  > 0

# 确认总数
grep -c "^\s*[A-Z][a-zA-Z0-9_]*\s*{" instr.rs  → ~97
```

---

## §7 实施顺序

| 步骤 | REQ | 内容 | 风险 | 状态 |
|------|-----|------|------|------|
| 1 | VR-003 | GprBinOp + GprCondAction 合并 | 🟢 安全 — 纯参数化 | ✅ 完成 |
| 2 | VR-002 | DotProduct 合并 | 🟢 安全 — dtype 参数化 | ✅ 完成 |
| 3 | VR-004 | Quant* 去重（删除复用已有指令的 10 个） | 🟢 安全 — 语义等价 | ✅ 完成 |
| 4 | VR-001 | QuantBlockLoad + QuantBiPlaneLoad 合并 | 🟡 中 — x86_lower 需重构 | ✅ 完成 |
| 5 | VR-005~010 | 缺失指令补全 | 🟢 安全 — 纯新增 | ✅ 完成 |
| 6 | VR-011 | auto_select 映射更新 | 🟡 中 — 必须 exhaustive | ✅ 完成 |
| 7 | VR-012 | ISA Lowering 重构 | 🟡 中 — 三后端同步 | ✅ 完成 |
| 8 | VR-013 | 验证 | — | ✅ 完成 |

---

## §8 修改文件总表

| 文件 | REQ | 改动 |
|------|-----|------|
| `codegen/vm/instr.rs` | VR-001~010 | 删除 41 变体 + 新增 13 变体 + 新增枚举类型 |
| `codegen/vm/auto_select.rs` | VR-011 | 更新所有 TraceOp→VmInstr 映射 |
| `codegen/vm/x86_lower.rs` | VR-012 | 删除旧 arm + 新增合并 arm + 新增缺失 arm |
| `codegen/vm/aarch64_lower.rs` | VR-012 | 同上 |
| `codegen/vm/gpu_lower.rs` | VR-012 | 同上 |
| `codegen/vm/plan_lower.rs` | VR-011 | 替换直接 emit GgufXxxLoad 为 QuantBlockLoad |
| `codegen/vm/opt_pass.rs` | VR-001~004 | 更新引用已删除变体的 pass |
| `codegen/vm/verify.rs` | VR-001~004 | 更新 def-before-use 规则中的变体名 |
| `codegen/vm/reg_alloc.rs` | VR-001~004 | 更新 referenced_vregs match arm |
| `codegen/vm/vm_state.rs` | VR-001~004 | 更新 VM interpreter（如有引用） |
| `codegen/vm/structural_builder.rs` | VR-001 | 更新结构型算子构建中的 GgufXxxLoad 引用 |
