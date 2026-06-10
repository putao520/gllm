# 量化算子 JIT 管线化 (QUANT-PIPELINE-JIT)

> **SSOT**: 本文件定义 QuantGather/QuantGemm 从"直接发射 VmInstr"迁移到"标量实现 → SymExec trace → auto_select → VmInstr → ISA lowering"完整 JIT 管线的架构。
>
> **核心铁律**: 禁止在 plan_lower.rs 中按 OpKind/QuantType 手写 VmInstr 发射函数。所有量化算子必须走完整 JIT 管线。
>
> **替代关系**: 本文件替代 plan_lower.rs 中 `emit_quant_gather_classic` / `emit_quant_gather_complex` / `emit_quant_gemm_tiled` 的直接 VmInstr 发射。
>
> **与 23-QUANT-CODEGEN-ALGO 的关系**: 23 定义量化算法与微核体系（QuantFormatDescriptor、三级路径、参数化微核模板、5 ISA × N 格式矩阵）；本文件定义如何将这些算子迁移到完整 JIT 管线（TraceOp 管线化）。23 管"算法设计"，本文件管"管线集成"。
>
> **状态**: ✅ 全部实现 — DecodeTraceBuilder 统一所有格式的解量化 trace 生成; auto_lower_trace_raw 驱动 VmInstr → ISA lowering; emit_quant_gather_trace_driven 替代手写 VmInstr 发射; REQ-QPJ-001~005 全部完成。

## §0 问题根因

### §0.1 当前架构的系统性缺陷

```
当前（半参数化，SPEC 23 已建立微核体系但 lowering 仍手写）:
  plan_lower.rs::emit_quant_gather_classic()
    → 直接构造 VmInstr::GgufInt4Load / GgufF16ScaleLoad / VecStore
    → elem = dtype.elem_bytes() 写死 4 (F32)
    → offset 计算假设 F32 布局
    → 每种量化格式需要单独手写 lowering
    → 人脑计算 offset/dtype → 系统性 NaN 根源
  plan_lower.rs::emit_quant_gemm_tiled()
    → 外层 M/N/K tiling 合理（保留）
    → 微核 prologue 仍手写 VmInstr 发射（需迁移到 trace 模板）

应该（JIT 铁律）:
  Scalar(Q4_0 dequant) → 参数化 trace 模板 → auto_select → VmInstr → ISA Lowering
    → dtype 从 QuantFormatDescriptor 自然传播
    → offset 按实际字节大小计算
    → 硬件差异由 ISA lowering 层处理
    → QuantGemm 外层 tiling 保留，微核 prologue 改为 trace 模板驱动
```

### §0.2 直接发射导致的具体 BUG

| BUG | 根因 |
|-----|------|
| Q4_0 QuantGather 全 NaN | offset/stride 计算假设 F32 字节大小 |
| Q3_K/Q4_K/Q5_K/Q6_K 全失败 | 每种格式的 decode 逻辑各自手写，无法保证正确性 |
| AWQ4/GPTQ4 zero-point 偏移错 | QuantFormatDescriptor 元数据未被 trace 管线消费 |
| 19GB scratchpad 过度分配 | buffer_alloc 中间张量写死 F32 elem_bytes |

### §0.3 违反的 CLAUDE.md 铁律

1. ❌ **禁止跳过 Lifting 阶段直接手写 ISA 汇编** — QuantGather/QuantGemm 直接写 VmInstr
2. ❌ **禁止创建 per-OpKind 手写函数** — `emit_quant_gather_classic` / `emit_quant_gather_complex` / `emit_quant_gemm_tiled` 都是 per-OpKind
3. ❌ **禁止以"结构型算子"为由绕过 auto_select** — QuantGather 被标记为 structural 绕过管线

## §1 架构设计

### §1.1 管线总览

```
QuantGather / QuantGemm
    │
    ▼
Phase 0: 标量实现 (scalar-ops/src/)
    scalar_q4_0_gather() / scalar_q8_0_gather() / ...
    scalar_q4_0_gemm() / scalar_q8_0_gemm() / ...
    → 正确的 block 解包 + scale + dequant，dtype 感知
    │
    ▼
Phase 0.5: 手动 trace 注入 (registry.rs)
    QuantFormatDescriptor 驱动的参数化 trace 模板
    → 不是全量 SymExec（GGUF block 结构太复杂），
      而是参数化 trace 模板（每族量化格式一个模板）
    │
    ▼
Phase 1: ComputePattern 分类 (semantic_dag.rs)
    QuantGather → Injective (多输入逐元素)
    QuantGemm → Gemm (矩阵乘)
    │
    ▼
Phase 2: auto_select (auto_select.rs)
    TraceOp::QuantBlockLoad → VmInstr::GgufF16ScaleLoad / GgufInt4Load / ...
    TraceOp::QuantDequant → VmInstr::GgufInt4Load + VecBinOp::Mul
    TraceOp::QuantStore → VmInstr::VecStore
    → dtype 自然传播，offset 正确计算
    │
    ▼
Phase 3: ISA Lowering (x86_lower.rs / aarch64_dynasm.rs / gpu_lower.rs)
    VmInstr::GgufInt4Load → vmovq + vpmovzxbd + vpand + vcvtdq2ps + vsubps
    VmInstr::GgufF16ScaleLoad → vmovd + vcvtph2ps + vbroadcastss
    → 硬件特化指令选择
```

### §1.2 关键设计决策

**Q: 为什么不全量 SymExec？**

A: GGUF block 解码涉及复杂的内存寻址（block_ptr + offset + nibble split），SymExec 无法追踪。解决方案：**参数化 trace 模板**。每族量化格式定义一个 trace 生成函数，参数化于 `QuantFormatDescriptor` 的元数据（block_size, block_bytes, data_kind, scale_layout 等）。这保留了管线的核心优势（dtype 传播、offset 正确、硬件特化），同时绕过 SymExec 的限制。

**Q: 参数化 trace 模板 vs 直接 VmInstr 发射的区别？**

A: 核心区别在于**抽象层级**：
- 直接发射：plan_lower.rs 手动管理寄存器分配、地址计算、内存布局 → 每种格式 × 每种硬件 × 每种 dtype 组合都要手写
- 参数化 trace：trace 模板描述"做什么"（load scale、unpack nibbles、multiply），auto_select 根据 hardware profile 选择"怎么做" → N 种格式 × M 种硬件 = N 个模板 + M 个 auto_select 规则，不是 N×M 个手写函数

### §1.3 新增 TraceOp 变体

```rust
// ── GGUF 量化块加载 ──

/// 加载量化 block 的 scale（f16/bf16 → broadcast compute_dtype）。
/// 参数: block_ptr slot, block 内偏移, QuantType（决定 scale 格式）
/// 映射: x86 GgufF16ScaleLoad / ARM scalar f16→compute_dtype / GPU ld.global.b16
TraceOp::QuantScaleLoad {
    block_ptr: u32,        // 指向当前 block 的指针 slot
    byte_offset: usize,    // scale 在 block 内的字节偏移
    quant_type: QuantType, // 决定 scale 的格式 (f16/bf16/f32/8bit)
},

/// 加载量化 block 的 packed data 并解包为 compute_dtype 向量。
/// 参数: block_ptr slot, data 在 block 内偏移, data_kind 决定解包方式
/// 映射: x86 GgufInt4Load/GgufInt4HighLoad/GgufInt8Load /
///       ARM scalar byte unpack / GPU ld.global + bit manipulation
TraceOp::QuantDataLoad {
    block_ptr: u32,          // 指向当前 block 的指针 slot
    byte_offset: usize,      // packed data 在 block 内的字节偏移
    data_kind: QuantDataKind, // PackedInt4/SignedPackedInt4/Int8/PackedInt5/PackedInt6/...
    pass: QuantLoadPass,     // LowNibble / HighNibble / Single (4-bit split layout)
},

/// 加载量化 block 的 zero-point / min / bias。
/// 参数: block_ptr slot, zp 在 block 内偏移, zero_layout 决定格式
TraceOp::QuantZeroLoad {
    block_ptr: u32,
    byte_offset: usize,
    zero_layout: ZeroLayout, // None/StaticBias(i32)/PackedInt4/PerChannelF16
},

/// 加载子块 scale（K-Quant 层级化 scale）。
TraceOp::QuantSubScaleLoad {
    block_ptr: u32,
    byte_offset: usize,
    bits: usize,         // 6-bit packed / 8-bit direct
    sub_block_size: usize,
},

/// 加载 high bits（INT5/INT6 的额外位）。
TraceOp::QuantHighBitsLoad {
    block_ptr: u32,
    byte_offset: usize,
    bits_per_elem: usize,
},

/// Codebook 查找（SqueezeLLM / IQ 系列）。
TraceOp::QuantCodebookDequant {
    indices: u32,
    codebook_ptr: u32,
    vector_size: usize,
    bits_per_entry: usize,
},
```

```rust
/// 4-bit 加载 pass（GGUF split layout）。
#[derive(Debug, Clone, Copy)]
pub enum QuantLoadPass {
    /// 低 nibble: (byte & 0x0F) - 8.0（有符号）或 (byte & 0x0F)（无符号）
    LowNibble,
    /// 高 nibble: ((byte >> 4) & 0x0F) - 8.0（有符号）或 ((byte >> 4) & 0x0F)（无符号）
    HighNibble,
    /// 单 pass（8-bit / 5-bit / 6-bit 等）
    Single,
}
```

### §1.4 参数化 trace 模板

每族量化格式定义一个**trace 生成函数**，接受 `QuantFormatDescriptor` 参数，输出 `Vec<TraceOp>` 序列。

```rust
/// 量化 decode trace 模板：给定 QuantFormatDescriptor，
/// 生成一个 block 的 decode trace（加载 scale → 加载 data → 解包 → 乘 scale → 输出）。
fn quant_block_decode_trace(
    desc: &QuantFormatDescriptor,
    block_ptr_slot: u32,  // 指向当前 block 的指针
    output_slot: u32,     // 输出向量 slot
    block_elem_offset: usize, // 当前 block 在输出中的元素偏移
) -> Vec<TraceOp>
```

**模板按 data_kind 分族**：

| data_kind | 模板 | 生成 TraceOps |
|-----------|------|--------------|
| `SignedPackedInt4` (Q4_0) | `signed_int4_decode_trace` | QuantScaleLoad → QuantDataLoad(LowNibble) → Mul → QuantDataLoad(HighNibble) → Mul → Store |
| `PackedInt4` (Q4_1) | `unsigned_int4_decode_trace` | QuantScaleLoad → QuantDataLoad(LowNibble) → Mul → QuantZeroLoad → Add → (HighNibble 同) → Store |
| `Int8` (Q8_0) | `int8_decode_trace` | QuantScaleLoad → QuantDataLoad(Single) → Mul → Store |
| `PackedInt5` (Q4_K) | `int5_decode_trace` | QuantSubScaleLoad → QuantDataLoad(Single) → QuantHighBitsLoad → 合并 → Mul → Store |
| `PackedInt6` (Q5_K) | `int6_decode_trace` | QuantSubScaleLoad → QuantDataLoad(Single) → QuantHighBitsLoad → 合并 → Mul → Store |
| `SuperLowBit` (TQ/IQ) | `superlowbit_decode_trace` | QuantScaleLoad → QuantCodebookDequant → Mul → Store |
| `Float4` (MXFP4/NVFP4) | `fp4_decode_trace` | QuantScaleLoad → QuantDataLoad(Single) → Cast(E2M1→F32) → Mul → Store |
| `Bfloat16` | `bf16_passthrough` | VecLoad → Cast(BF16→F32) → Store |
| `Float16` | `fp16_passthrough` | VecLoad → Cast(F16→F32) → Store |
| `Float32` | `f32_passthrough` | VecLoad → Store |

### §1.5 auto_select 扩展

```rust
// auto_select.rs dispatch_trace_op 新增 match arms:

TraceOp::QuantScaleLoad { block_ptr, byte_offset, quant_type } => {
    match quant_type {
        QuantType::Q4_0 | QuantType::Q8_0 | ... => {
            // x86: VmInstr::GgufF16ScaleLoad
            // ARM: 逐字节构建 f16 → f32
            // GPU: ld.global.b16 + cvt.rn.f32.f16
            prog.emit(VmInstr::GgufF16ScaleLoad { dst, base: slots[*block_ptr], offset, width })
        }
        QuantType::Q4K | QuantType::Q5K | ... => {
            // 层级化 scale: 可能需要多个加载步骤
        }
        _ => ...
    }
}

TraceOp::QuantDataLoad { block_ptr, byte_offset, data_kind, pass } => {
    match (data_kind, pass) {
        (SignedPackedInt4, LowNibble) => VmInstr::GgufInt4Load,
        (SignedPackedInt4, HighNibble) => VmInstr::GgufInt4HighLoad,
        (PackedInt4, LowNibble) => VmInstr::GgufUInt4Load,
        (PackedInt4, HighNibble) => VmInstr::GgufUInt4HighLoad,
        (Int8, Single) => VmInstr::GgufInt8Load,
        ...
    }
}
```

## §2 标量实现

### §2.1 QuantGather 标量实现矩阵

每种量化格式族一个标量 gather 函数（不是每种格式一个——同族格式共享）：

| 标量函数 | 覆盖格式 | 关键差异 |
|---------|---------|---------|
| `scalar_q4_0_gather` | Q4_0 | SignedPackedInt4, 单 scale f16, 无 zero-point |
| `scalar_q4_1_gather` | Q4_1 | PackedInt4(无符号), scale f16 + min f16 |
| `scalar_q8_0_gather` | Q8_0, Q8_1 | Int8, 单 scale f16, Q8_1 有 sum f16 |
| `scalar_kquant_gather` | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | 层级化 scale, 2/3/4/5/6-bit packed |
| `scalar_mxfp4_gather` | MXFP4 | E2M1 + E8M0 scale |
| `scalar_awq4_gather` | AWQ4 | 行优先, FP16 zero-point |
| `scalar_gptq4_gather` | GPTQ4 | 列交织, INT4 packed zp + 1 偏移 |

**标量函数签名统一**：
```rust
// E: Element — 输出元素类型, 匹配 compute_dtype (由 TensorMeta.dtype × DeviceProfile 推导)
pub extern "C" fn scalar_<format>_gather(
    indices: *const u32,       // token IDs (整数索引, 非浮点)
    table_quant: *const u8,    // 量化 embed table
    output: *mut E,            // compute_dtype 输出
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
)
```

标量函数内部用 `QuantFormatDescriptor` 的元数据（block_size, block_bytes, scale_layout 等）驱动解码，不硬编码任何偏移。

### §2.2 QuantGemm 标量实现

QuantGemm 的标量实现不需要 SymExec 全量 trace（GEMM 循环结构由 emit_gemm_inline 统一处理）。关键改变是**微核 prologue 从手写 VmInstr 改为 trace 模板驱动**：

```rust
// 当前（错误）: emit_quant_gemm_tiled 直接发射 GgufInt4Load + VecBinOp::Mul
// 正确: 微核 prologue 调用 quant_block_decode_trace 生成 TraceOps → auto_select 映射
```

GEMM 的外层循环（M/N/K tiling）仍由 `emit_gemm_inline` 统一处理，但**每个 K tile 内的权重 decode** 从直接发射改为 trace 模板。

### §2.3 buffer_alloc 修复

```rust
// 当前（错误）:
let elem_bytes = if tensor.producer.is_some() {
    DType::F32.size_bytes()  // BUG: 写死 F32，应从 producer 的输出 dtype 推导
} else {
    tensor.dtype.size_bytes()
};

// 正确: 从 producer 的 ComputePattern 推导输出 dtype
let elem_bytes = if let Some(producer) = tensor.producer {
    producer_output_dtype(producer, graph).elem_bytes()
} else {
    tensor.dtype.size_bytes()
};

// 正确: 从 tensor 的实际 dtype 推导
let elem_bytes = tensor.dtype.size_bytes();
// 如果 producer 的输出 dtype 与输入不同（如 QuantGather: Q4_0 输入 → F32 输出），
// 使用 producer 的输出 tensor dtype
```

## §3 注册表与 Trace 注入

### §3.1 QuantGather 注册

```rust
// registry.rs

// Q4_0 族: SignedPackedInt4, 单 scale, 无 zp
reg.register(OpKindKey::QuantGather, q4_0_gather_sig);
reg.inject_trace(OpKindKey::QuantGather, OpTrace {
    op_kind: OpKind::QuantGather { quant_type: Q4_0, ... },
    pattern: ComputePattern::Injective {
        body: quant_block_decode_trace(&Q4_0_DESCRIPTOR, ...),
        num_inputs: 2,
    },
    signature: q4_0_gather_sig,
});
```

### §3.2 QuantGemm 注册

```rust
// QuantGemm 已注册为 Gemm pattern
// 改变: 微核 prologue 从直接 VmInstr 发射改为 trace 模板
// emit_gemm_inline 检测 quant_type → 调用 quant_block_decode_trace 生成 K-tile 内循环
```

### §3.3 计算模式分类

| OpKind | ComputePattern | 处理路径 |
|--------|---------------|---------|
| QuantGather | Injective { body, num_inputs: 2 } | emit_injective_inline + auto_lower_trace |
| QuantGemm | Gemm | emit_gemm_inline + quant_block_decode_trace for K-tile |

## §4 auto_select 扩展

### §4.1 新增 dispatch 规则

```rust
// auto_select.rs

// QuantScaleLoad: 根据 quant_type 选择 scale 加载方式
TraceOp::QuantScaleLoad { block_ptr, byte_offset, quant_type } => {
    let r = alloc_slot(dst_slot);
    match quant_type {
        // f16 scale → GgufF16ScaleLoad (vmovd + vcvtph2ps + vbroadcastss)
        Q4_0 | Q4_1 | Q8_0 | Q8_1 => {
            prog.emit(VmInstr::GgufF16ScaleLoad { dst: r, base: slots[*block_ptr], offset: Const(*byte_offset), width });
        }
        // 层级化 scale → QuantSubScaleLoad + unpack
        Q4K | Q5K | Q6K => { ... }
        // E8M0 block scale → MXFP decode
        Mxfp4 { .. } => { ... }
        _ => ...
    }
}

// QuantDataLoad: 根据 data_kind + pass 选择解包方式
TraceOp::QuantDataLoad { block_ptr, byte_offset, data_kind, pass } => {
    let r = alloc_slot(dst_slot);
    match (data_kind, pass) {
        (SignedPackedInt4, LowNibble) => {
            prog.emit(VmInstr::GgufInt4Load { dst: r, base: slots[*block_ptr], offset, width });
        }
        (SignedPackedInt4, HighNibble) => {
            prog.emit(VmInstr::GgufInt4HighLoad { dst: r, base: slots[*block_ptr], offset, width });
        }
        (PackedInt4, LowNibble) => {
            prog.emit(VmInstr::GgufUInt4Load { dst: r, base: slots[*block_ptr], offset, width });
        }
        (PackedInt4, HighNibble) => {
            prog.emit(VmInstr::GgufUInt4HighLoad { dst: r, base: slots[*block_ptr], offset, width });
        }
        (Int8, Single) => {
            prog.emit(VmInstr::GgufInt8Load { dst: r, base: slots[*block_ptr], offset, width });
        }
        (PackedInt5, Single) => { /* INT5 解包 */ }
        (PackedInt6, Single) => { /* INT6 解包 */ }
        (SuperLowBit, Single) => { /* codebook lookup */ }
        (Float4, Single) => { /* E2M1 decode */ }
        _ => Err(...)
    }
}
```

## §5 迁移计划

### §5.1 阶段划分

**Phase 1: TraceOp 扩展 + auto_select 基础** (REQ-QPJ-001)
- 在 trace.rs 新增 QuantScaleLoad / QuantDataLoad / QuantZeroLoad / QuantSubScaleLoad / QuantHighBitsLoad / QuantCodebookDequant 变体
- 在 auto_select.rs 新增对应 match arms，映射到现有 VmInstr
- 不改变现有 QuantGather/QuantGemm lowering 路径，只建立基础设施

**Phase 2: QuantGather trace 模板化** (REQ-QPJ-002)
- 为 Q4_0/Q4_1/Q8_0/Q8_1 创建参数化 trace 模板
- registry.rs 注入 trace（替代当前空 trace）
- plan_lower.rs 中 `dispatch_structural` 的 QuantGather 分支改为调用 `emit_injective_inline` + `auto_lower_trace`
- 删除 `emit_quant_gather_classic`
- 验证: Q4_0/Q8_0 E2E 测试通过

**Phase 3: QuantGather 复杂格式 trace 模板化** (REQ-QPJ-003)
- 为 Q2_K/Q3_K/Q4_K/Q5_K/Q6_K 创建 trace 模板
- 为 AWQ4/GPTQ4/MXFP4/NVFP4/TQ 创建 trace 模板
- 删除 `emit_quant_gather_complex`
- 验证: 所有量化格式 E2E 测试通过

**Phase 4: QuantGemm 微核 trace 模板化** (REQ-QPJ-004)
- 将 `emit_quant_gemm_tiled` 的微核 prologue 改为 trace 模板驱动
- 保留外层 M/N/K tiling 框架（这是合理的 GEMM 结构，不需要改变）
- 微核内部的 scale load / data load / dequant / FMA 改为 TraceOp → auto_select
- 删除手写的 GgufInt4Load/GgufF16ScaleLoad 发射代码
- 验证: 所有量化格式 GEMM 正确性

**Phase 5: 清理** (REQ-QPJ-005)
- 删除旧版 `emit_quant_gather_classic` / `emit_quant_gather_complex`（Phase 2/3 已用 trace 模板替代）
- 重构 `emit_quant_gemm_tiled` → `emit_gemm_quant`：保留外层 M/N/K tiling 框架（Phase 4 已将 prologue 改为 trace 模板），删除残留的手写 prologue 代码
- 删除 buffer_alloc.rs 中写死的 `DType::F32.size_bytes()`
- 删除 mega_kernel.rs 中所有诊断 eprintln
- 验证: 全量 E2E 测试通过

### §5.2 验证标准

| 阶段 | 验证命令 | 通过标准 |
|------|---------|---------|
| Phase 1 | `cd gllm-kernels && cargo check` | 编译通过 |
| Phase 2 | `cargo test --test test_e2e_generator -- gguf_q4_0 --test-threads=1` | 输出有意义文本（非 "!!!!!!!!!!"） |
| Phase 2 | `cargo test --test test_e2e_generator -- gguf_q8_0 --test-threads=1` | 通过 |
| Phase 3 | `cargo test --test test_e2e_generator --test-threads=1` | 所有量化格式通过 |
| Phase 4 | `cd gllm-kernels && cargo test --lib` | 所有单元测试通过 |
| Phase 5 | `cargo test --test test_e2e_generator --test-threads=1` | 全量 E2E 通过 |

## §6 REQ 清单

| REQ ID | 标题 | 描述 |
|--------|------|------|
| REQ-QPJ-001 | TraceOp 量化扩展 | 新增 QuantScaleLoad/QuantDataLoad/QuantZeroLoad/QuantSubScaleLoad/QuantHighBitsLoad/QuantCodebookDequant TraceOp 变体 + auto_select 映射 |
| REQ-QPJ-002 | QuantGather Classic 管线化 | Q4_0/Q4_1/Q8_0/Q8_1 的 trace 模板 + registry 注入 + dispatch_structural 改为 auto_select |
| REQ-QPJ-003 | QuantGather Complex 管线化 | K-Quant/AWQ/GPTQ/MXFP4/NVFP4/TQ 的 trace 模板 |
| REQ-QPJ-004 | QuantGemm 微核管线化 | 微核 prologue 改为 trace 模板，外层 tiling 保留 |
| REQ-QPJ-005 | 清理与修复 | 删除手写 lowering 函数 + buffer_alloc 修复 + 诊断代码清理 |

## §7 修改文件总表

### gllm-kernels

| 文件 | 修改 |
|------|------|
| `src/compiler/trace.rs` | 新增 6 个 TraceOp 变体 |
| `src/compiler/codegen/vm/auto_select.rs` | 新增 6 个 dispatch_trace_op match arm |
| `src/compiler/registry.rs` | QuantGather/QuantGemm trace 注入 |
| `src/compiler/codegen/vm/plan_lower.rs` | 删除 emit_quant_gather_classic/complex + QuantGemm 微核改为 trace 模板 |
| `src/compiler/buffer_alloc.rs` | elem_bytes 改为从 tensor.dtype 推导 |
| `scalar-ops/src/gather.rs` | 新增 Q4_1/Q8_0/K-Quant 标量 gather |

### gllm

| 文件 | 修改 |
|------|------|
| `src/engine/mega_kernel.rs` | 删除诊断 eprintln（66+ 行） |
