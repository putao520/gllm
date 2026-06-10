# 全量化格式原生 JIT 生成器 (QUANT-CODEGEN-ALGO)

> **实现状态**: ✅ REQ-QCG-001~013 全部完成 — QuantFormatDescriptor 33 格式注册; DotProduct 5 DotDtype; QuantGemmPlan::derive 参数化微核 (Native/Assisted/HighBitMerge/DequantFma); Int8Native VPDPBUSD/SDOT; Assisted nibble unpack+FMA; QuantGather 算子; 量化感知融合; DecodeTraceBuilder 统一全格式 trace 生成; 🔄 Weight-Only GEMM 寄存器内反量化优化 + dtype 转换 JIT 化为未来性能工作
>
> **SSOT**: 本文件定义"每种量化格式 × 每种算子 → 对应硬件原生指令机器码"的原生量化 JIT 体系。
>
> **核心铁律**：禁止 "decode quant → F32 → FMA" 的欺骗式中间转换。所有量化算子必须直接在 packed quantized data 上用硬件原生指令计算。量化格式是 codegen 的**一等参数**，与硬件 profile 一起决定最终机器码。
>
> **替代关系**：本文件完全取代原 SPEC/23 的 DecodeTraceBuilder "解量化 + F32 FMA" 方案。
>
> **与 24-QUANT-PIPELINE-JIT 的关系**：本文件定义量化算法与微核体系（§1-§5）和 REQ（§6）；24-QUANT-PIPELINE-JIT 定义如何将 QuantGather/QuantGemm 从"直接 VmInstr 发射"迁移到"TraceOp 管线化"（标量实现 → trace 模板 → auto_select → VmInstr → ISA lowering）。两文件互补，不重叠。
>
> **实现状态**: ✅ REQ-QCG-001/002/003/004/005/006/007/008/009/010/011/012/013 已完成 — QuantFormatDescriptor (33 格式) + DotProduct VmInstr (5 DotDtype) + QuantGemmPlan 参数化微核模板 + Int8Native 路径 (VNNI/SDOT) + Assisted 路径 (nibble unpack + FMA) + HighBitMerge INT5/INT6 路径 (Q5_0/Q5_1/Q5_K/Q6_K) + Q2K/Q3K 极低比特路径 (Hierarchical scale/zero) + IQ Codebook 路径 (9 IQ 格式 + QuantCodebookLookup VmInstr) + FP4 路径 (MXFP4/NVFP4 E2M1 LUT + SM100 tcgen05 原生) + AWQ4/GPTQ4 Weight-Only 路径 ((qw-zp)×scale + 张量分离存储) + QuantGather + 量化感知融合 + 量化权重原格式直传 JIT (zero-copy weight path); 🔄 Weight-Only GEMM 寄存器内反量化优化 + dtype 转换场景 JIT 化为未来性能工作
>
> 交叉引用: `37-HARDWARE-ACCELERATION.md` 定义了 NVFP4/MXFP4 硬件原生路径（SM100 tcgen05.mma .f4）和动态精度热切换（REQ-HWACC-007）。22 种 QuantType 的硬件加速映射见该文档。

## §0 QuantType 全量清单

本 SPEC 覆盖的量化格式按 §2.2 `QuantDataKind` 分族：

| 族 | 格式 | QuantDataKind | REQ |
|----|------|--------------|-----|
| **原生浮点** | F32, BF16, FP16 | — (无 scale/zero) | — |
| **INT8** | Q8_0, Q8_1 | `Int8` | REQ-QCG-004 |
| **INT8 K-Quant** | Q8_K | `Int8` (bsums 偏置消除) | REQ-QCG-004 |
| **INT4** | Q4_0, Q4_1 | `PackedInt4` / `SignedPackedInt4` | REQ-QCG-005 |
| **INT4 K-Quant** | Q4_K | `PackedInt4` (多级 scale) | REQ-QCG-005 |
| **INT5** | Q5_0, Q5_1 | `PackedInt5` (qh 高位合并) | REQ-QCG-006 |
| **INT5 K-Quant** | Q5_K | `PackedInt5` (多级 scale) | REQ-QCG-006 |
| **INT6 K-Quant** | Q6_K | `PackedInt6` | REQ-QCG-006 |
| **INT2 K-Quant** | Q2_K | `PackedInt2` (跨步解包) | REQ-QCG-007 |
| **INT3 K-Quant** | Q3_K | `PackedInt3` (hmask+qs 组合) | REQ-QCG-007 |
| **IQ 1-bit** | IQ1_S, IQ1_M | `SuperLowBit` (A8/E8 lattice) | REQ-QCG-008 |
| **IQ 2-bit** | IQ2_S, IQ2_XS, IQ2_XXS | `SuperLowBit` (bit 域解包) | REQ-QCG-008 |
| **IQ 3-bit** | IQ3_S, IQ3_XS, IQ3_XXS | `SuperLowBit` (grid LUT) | REQ-QCG-008 |
| **IQ 4-bit** | IQ4_NL, IQ4_XS | `SuperLowBit` | REQ-QCG-008 |
| **FP4** | MXFP4, NVFP4 | `Float4` / `Nvfp4` | REQ-QCG-009/009a |
| **Weight-Only** | AWQ4, GPTQ4 | `PackedInt4` (张量分离) | REQ-QCG-010/010a |
| **3-bit** | SqueezeLLM | `SuperLowBit` (线性 3-bit) | REQ-QCG-008 |
| **Tesla TQ** | TQ1_0, TQ2_0 | `SuperLowBit` (Tesla-optimized) | REQ-QCG-008 |

**总计**: 3 原生浮点 + 3 INT8 + 3 INT4 + 3 INT5/6 + 2 INT2/3 + 10 IQ + 2 FP4 + 2 Weight-Only + 1 SqueezeLLM + 2 TQ = **31 种 QuantType**（跨 7 个 data_kind 族）。

> **注**: 旧版文档中"22 种 QuantType"的计数已过时。当前清单包含实际模型文件中出现的全部 GGUF/HF 量化格式。如新增格式，更新本表即可。

## §1 核心设计原则

### §1.1 原生量化计算 (Native Quantized Compute)

```
错误（欺骗式）：加载 quant block → 解码到 F32 向量 → F32 FMA 累加
正确（原生）：加载 packed quant data → 硬件原生 dot-product 指令 → int32/fp32 累加
```

每种算子的 codegen 函数签名必须包含 `quant_type: QuantType` 参数（可选，None=F32/BF16 原生路径）。codegen 根据 `(quant_type, device_profile)` 选择硬件原生指令路径：

| 硬件 | BF16 路径 | FP16 路径 | F32 路径 | INT8 路径 | INT4 路径 | FP4 路径 |
|------|----------|----------|---------|----------|----------|---------|
| x86 VNNI+BF16 | VDPBF16PS (bf16×bf16→f32) | N/A | VFMADD (f32×f32→f32) | VPDPBUSD (i8×i8→i32) | 软件展开 4-bit + VNNI | N/A |
| x86 AMX | TDPBF16PS (bf16×bf16→f32) | N/A | 标量 FMA | TDPBSSD (i8×i8→i32) | 软件展开 + TDPBSSD | N/A |
| ARM SVE2/SME2 | BFMMLA (bf16×bf16→f32) | FMMLA (fp16×fp16→f32) | FMLA (f32×f32→f32) | SDOT (i8×i8→i32) | 软件展开 + SDOT | N/A |
| SM80 | HMMA (bf16/fp16 tensor core) | HMMA fp16 | 标量 FMA | m16n8k32.s32.u8.u8 | 软件展开 + IMMA | N/A |
| SM90 | WGMMA bf16/fp16 | WGMMA fp16 | 标量 FMA | m64nNk32.f32.int8 | 软件展开 + INT8 TC | N/A |
| SM100+ | WGMMA bf16/fp16 | WGMMA fp16 | 标量 FMA | — | — | tcgen05 FP4 原生 |
| GFX12 | WMMA bf16 | WMMA fp16 | 标量 FMA | v_wmma_i32_16x16x16_iu4_iu8 | 4-bit×8-bit 原生 | N/A |

**BF16/FP16 是量化框架的一等公民**：它们不需要 scale/zero/unpack，直接在 packed float data 上用硬件原生指令计算。在统一框架中，BF16/FP16 的微核 prologue 是空操作（无 scale/zero），compute 直接用 VDPBF16PS/BFMMLA/HMMA 等指令，epilogue 无需缩放。

**关键**：即使硬件没有 native 4-bit 指令，codegen 也必须在**一个微核内**完成 "unpack nibbles → int8 dot product"，不做中间 F32 转换。累加在 INT32，最后一步才缩放到 FP32。

### §1.2 量化格式作为一等参数

```rust
// 每个 emit 函数都接受 quant_type 参数（F32/BF16 也是合法值）
fn emit_gemm_inline(
    prog: &mut VmProgram,
    m: BoundExpr, n: usize, k: usize,
    quant_type: QuantType,  // ← F32/BF16/FP16 也是合法值，统一入口
    width: SimdWidth,
    input_ptr: VRegId, weight_ptr: VRegId, output_ptr: VRegId,
    dtype: QuantPrecision,
) -> Result<(), CompilerError>
```

`quant_type` 影响：
1. **权重的读取步长**：`block_bytes`（BF16=2, F32=4, Q4_0=18, Q8_0=34...）
2. **计算指令选择**：VDPBF16PS vs VPDPBUSD vs VFMADD vs WMMA
3. **累加器类型**：INT32（整数量化路径）vs FP32（浮点路径）
4. **缩放**：整型路径 int32_acc × scale → fp32；浮点路径无缩放

### §1.3 三级路径体系

```
Level 1: 硬件原生 (Hardware-Native)
  条件: 硬件有对应 bit-width/type 的 dot-product 指令
  示例: VDPBF16PS (bf16), BFMMLA (bf16), VPDPBUSD (int8), GFX12 WMMA a4w8 (int4×int8), SM90 IMMA (int8)
  特征: packed data → 一条指令 → fp32/int32 累加, 零软件解包

Level 2: 硬件辅助 (Hardware-Assisted)
  条件: 硬件有 int8 dot-product，但无 int4 指令
  示例: 4-bit nibbles 展开 → sign-extend → VPDPBUSD
  特征: 微核内完成展开+点积, 累加仍在 INT32

Level 3: 标量回退 (Scalar Fallback — 仅用于无 SIMD 的极端场景)
  条件: 无任何 SIMD 指令
  示例: 纯标量循环
  特征: 仍然是 JIT 生成, 不 fallback 到 Rust
```

**量化模型最终 Fallback (DequantFMA)**：仅限量化模型（quant_type = Some）。当硬件既无 INT8 dot-product 也无足够 SIMD 宽度执行标量 INT8 循环时（例如仅有 SSE2 的老 CPU），保留 decode→F32→FMA 作为量化模型的最终兜底。此路径**仅用于量化模型在无原生指令设备上的兼容运行**，非量化模型（F32/BF16）不走此路径。

```
DequantFMA 触发条件（全部满足才走）:
  1. 当前算子是量化模型算子（quant_type = Some(Q4_0/Q8_0/...)）
  2. 硬件无 VNNI / SDOT / IMMA / WMMA 等任何 dot-product 指令
  3. 硬件 SIMD 宽度 ≤ 128-bit (SSE2 只有 128-bit)
  4. 无法有效执行标量 INT8 展开

DequantFMA 路径:
  packed quant → decode block → F32 向量 → F32 FMA 累加
  这仍然是 JIT 生成的机器码，不 fallback 到 Rust

量化模型路径优先级: Native > Assisted > Scalar > DequantFMA(最终兜底)
非量化模型路径: 标准 F32/BF16 GEMM，不涉及量化路径选择
禁止: 在有 VNNI/SDOT/IMMA 的设备上对量化模型走 DequantFMA
```

## §2 QuantFormatDescriptor — 量化格式元数据

### §2.1 数据结构

保留 `QuantFormatDescriptor` 描述格式布局，但**不再用于生成 decode→F32 trace**，而是用于：

1. **计算权重地址步长**（`block_bytes`）
2. **确定展开策略**（4-bit 需要 nibble unpack，8-bit 直接使用）
3. **确定 scale/zero-point 加载方式**（影响微核 prologue）

```rust
#[derive(Debug, Clone)]
pub struct QuantFormatDescriptor {
    pub name: &'static str,
    pub quant_type: QuantType,
    pub block_size: usize,       // 每 block 多少 element (32/256)
    pub block_bytes: usize,      // 每 block 多少 byte
    pub bits_per_element: u8,    // 1/2/3/4/5/6/8

    /// Scale 布局（决定微核 prologue 如何加载 scale）
    pub scale_layout: ScaleLayout,
    /// Zero-point 布局
    pub zero_layout: ZeroLayout,
    /// 量化数据在 block 中的偏移
    pub data_offset: usize,
    /// 量化数据类型（决定硬件指令选择）
    pub data_kind: QuantDataKind,
    /// 可选 codebook（IQ 系列）
    pub codebook: Option<CodebookSpec>,
}

/// 数据的计算分类 — 决定使用哪种硬件指令
/// BF16/FP16/F32 是量化框架的一等公民，与整数量化格式使用同一个微核模板
#[derive(Debug, Clone, Copy)]
pub enum QuantDataKind {
    // === 原生浮点路径（无 scale/zero，无 unpack） ===
    /// BF16 — 直接使用 VDPBF16PS/BFMMLA/HMMA 等原生指令
    Bfloat16,
    /// FP16 — 直接使用 FMMLA/HMMA 等原生指令 (x86 无原生 FP16 计算)
    Float16,
    /// F32 — 标准 FMA 路径 (所有硬件都支持)
    Float32,

    // === 整数量化路径（需要 scale/zero + unpack） ===
    /// 有符号 8-bit 整数 → VPDPBUSD/SDOT/IMMA
    Int8,
    /// 无符号 4-bit packed → 需要 nibble unpack → 然后走 int8 dot-product
    /// 或者 GFX12 a4w8 直接原生
    PackedInt4,
    /// 有符号 4-bit (Q4_0 减 8) → 需要 sign-extend → int8 dot-product
    SignedPackedInt4,
    /// 5-bit (需要特殊展开)
    PackedInt5,
    /// 6-bit (需要特殊展开)
    PackedInt6,
    /// FP4 (e2m1) → 需要 FP4 tensor core 或软件模拟
    Float4,
    /// 1-3 bit 极低比特 (需要 codebook lookup → 然后走 dot-product)
    SuperLowBit,
}
```

### §2.2 格式注册表

保留 `QuantFormatRegistry`，每个 `QuantType` 注册一个 `QuantFormatDescriptor`。新增 `data_kind` 字段替代原来的 `DataLayout` 复杂枚举。

**BF16/FP16/F32 的注册条目**：
```rust
// BF16 — 原生浮点，无 scale/zero，block_size=1，block_bytes=2
QuantFormatDescriptor {
    name: "BF16", quant_type: QuantType::Bf16,
    block_size: 1, block_bytes: 2, bits_per_element: 16,
    scale_layout: ScaleLayout::None,
    zero_layout: ZeroLayout::None,
    data_offset: 0,
    data_kind: QuantDataKind::Bfloat16,
    codebook: None,
}
// FP16 — 原生浮点，无 scale/zero，block_size=1，block_bytes=2
QuantFormatDescriptor {
    name: "FP16", quant_type: QuantType::F16,
    block_size: 1, block_bytes: 2, bits_per_element: 16,
    scale_layout: ScaleLayout::None,
    zero_layout: ZeroLayout::None,
    data_offset: 0,
    data_kind: QuantDataKind::Float16,
    codebook: None,
}
// F32 — 基线浮点，block_size=1，block_bytes=4
QuantFormatDescriptor {
    name: "F32", quant_type: QuantType::F32,
    block_size: 1, block_bytes: 4, bits_per_element: 32,
    scale_layout: ScaleLayout::None,
    zero_layout: ZeroLayout::None,
    data_offset: 0,
    data_kind: QuantDataKind::Float32,
    codebook: None,
}
```

**NVFP 家族注册条目**：

NVFP 是 NVIDIA 定义的硬件原生浮点低比特格式族。与 AWQ/GPTQ 的"整数量化+软件反量化"不同，NVFP 直接被 Tensor Core 硬件解码，零软件开销。

| 格式 | 位宽 | block_size | block_bytes | 缩放模型 | data_kind | 硬件支持 |
|------|------|-----------|-------------|---------|-----------|---------|
| NVFP4 | 4-bit E2M1 | 64 | 36 | 三级: global_f32 × sub_block_UE4M3 × E2M1_lut | `Nvfp4` | SM100+ tcgen05 原生; SM<100 软件查表 |
| MXFP4 | 4-bit E2M1 | 32 (可配) | block_size/2 + 1 | 单级: E8M0 per block (`scale = 2^(byte-127)`) | `Float4` | 无硬件原生; 软件查表 |

NVFP4 注册:
```rust
QuantFormatDescriptor {
    name: "NVFP4", quant_type: QuantType::Nvfp4,
    block_size: 64, block_bytes: 36, bits_per_element: 4,
    scale_layout: ScaleLayout::SubBlock { count: 4, bytes: 4 },  // 4 × UE4M3 per block
    zero_layout: ZeroLayout::None,
    data_offset: 4,  // scales 在前 4 字节
    data_kind: QuantDataKind::Nvfp4,
    codebook: None,
}
```

MXFP4 注册:
```rust
QuantFormatDescriptor {
    name: "MXFP4", quant_type: QuantType::Mxfp4 { block_size: 32 },
    block_size: 32, block_bytes: 17, bits_per_element: 4,  // 1 byte E8M0 + 16 bytes packed
    scale_layout: ScaleLayout::ExternalArray { stride: 1, scale_dtype: ScaleDType::E8M0 },
    zero_layout: ZeroLayout::None,
    data_offset: 1,  // E8M0 scale 在第 1 字节
    data_kind: QuantDataKind::Float4,
    codebook: None,
}
```

NVFP 与 MXFP4 共享 E2M1 编码（16 值真值表），差异仅在缩放模型:
- NVFP4: 三级缩放（global_f32 × sub_block_UE4M3 × E2M1_lut），硬件原生在 Tensor Core 内部完成
- MXFP4: 单级缩放（E8M0 per block），软件查表解码

**AWQ/GPTQ 家族注册条目**：

AWQ (Activation-Aware Weight Quantization) 和 GPTQ (Hessian-Based Post-Training Quantization) 都是 Weight-Only 整数量化格式。反量化公式统一: `FP16_weight = (qw - zero_point) × scale`。

| 格式 | 位宽 | group_size | storage | zero_point 类型 | 打包顺序 | 硬件路径 |
|------|------|-----------|---------|----------------|---------|---------|
| AWQ4 | 4-bit | 128 | 张量分离 (独立 qweight/scales/zeros tensor) | FP16 (每 group 一个) | 行优先连续 nibble | 软件反量化→FP16→HMMA |
| GPTQ4 | 4-bit | 128 | 张量分离 (独立 qweight/scales/zeros tensor) | INT4 packed (8 个 zp 打包进 u32, 需 +1 偏移) | 列交织 (stride-16, Warp 合并访存) | 软件反量化→FP16→HMMA |

AWQ4 注册:
```rust
QuantFormatDescriptor {
    name: "AWQ4", quant_type: QuantType::AWQ4,
    block_size: 128, block_bytes: 72, bits_per_element: 4,
    // 72 bytes = 128×4bit/8 = 64 bytes qweight + 2 bytes f16 scale + 2 bytes f16 zero + 4 padding
    scale_layout: ScaleLayout::PerGroup { group_size: 128, dtype: F16 },
    zero_layout: ZeroLayout::PerGroup { group_size: 128, dtype: F16 },
    data_offset: 0,
    data_kind: QuantDataKind::PackedInt4,
    codebook: None,
}
```

GPTQ4 注册:
```rust
QuantFormatDescriptor {
    name: "GPTQ4", quant_type: QuantType::GPTQ4,
    block_size: 128, block_bytes: 72, bits_per_element: 4,
    scale_layout: ScaleLayout::PerGroup { group_size: 128, dtype: F16 },
    zero_layout: ZeroLayout::PerGroupPackedInt4 { group_size: 128, offset: 1 },  // +1 偏移补偿
    data_offset: 0,
    data_kind: QuantDataKind::PackedInt4,
    codebook: None,
}
```

**AWQ vs GPTQ 关键差异**:
- **打包顺序**: AWQ 行优先连续 (8 个连续权重打包进 u32)；GPTQ 列交织 (8 个间隔 stride=16 的权重打包进 u32，利于 Warp 合并访存)
- **Zero-point**: AWQ 直接 FP16；GPTQ INT4 打包 +1 偏移补偿
- **GGUF 原生支持**: GGUF 文件可直接包含 AWQ/GPTQ 量化权重，gllm 加载后走原生 AWQ4/GPTQ4 反量化路径（寄存器内 `(qw - zp) × scale`），不转换为其他格式
- **HuggingFace safetensors**: 同样走原生反量化路径，需理解张量分离存储布局

**SqueezeLLM 注册条目**:

SqueezeLLM 原始论文使用 3-bit 非均匀聚类量化(8 个聚类中心,per-tensor codebook)。gllm 对外发布的工程化版本采用**简化的线性 3-bit** 量化,与原论文有数学上的偏离但保留 3-bit 存储密度:

- 块结构:`[scale: f16 (2B) | qs: u8 [128]] = 130 bytes per 256 元素 block`(每 byte 持有 2 个 nibble,但只用低 3-bit)
- 解码公式:`value = scale × (q3 - 4.0)`,其中 `q3 ∈ [0, 7]`(等价于 zero-point=4 的对称线性 3-bit)
- 与 SqueezeLLM 论文的差异:per-tensor codebook 退化为均匀 3-bit 网格,牺牲精度换取实现简化

```rust
QuantFormatDescriptor {
    name: "Squeeze", quant_type: QuantType::Squeeze,
    block_size: 256, block_bytes: 130, bits_per_element: 3,
    scale_layout: ScaleLayout::BlockScalar { offset_bytes: 0, dtype: ScaleDType::F16 },
    zero_layout: ZeroLayout::StaticBias { value: 4 },  // q3 - 4 (对称线性)
    data_layout: DataLayout::PackedNibbles { offset: 2, low_first: true },
    data_kind: QuantDataKind::SuperLowBit,
    codebook: None,  // 线性 3-bit,无 codebook LUT
}
```

反量化: `value = scale × (q3 - 4.0)`(线性 3-bit + 对称 zero-point 偏移 4)。

未来如要恢复 SqueezeLLM 原论文的 codebook 路径,需:
1. 从权重文件加载 per-tensor 8 值 FP16 codebook(SqueezeLLM HF 模型在独立 tensor 中提供)
2. 注册 `data_kind: QuantDataKind::Codebook`(新增枚举)
3. JIT 中用 `CodebookLookup` VmInstr 替代 `q3 - 4.0` 的线性解码

## §3 参数化量化微核体系

### §3.1 核心架构：一个微核模板，参数驱动实例化

**铁律：量化微核不是 N 个手写函数，而是一个参数化模板。** `(QuantType × DeviceProfile)` 的每个组合自动实例化为对应机器码，不需要为每种硬件写独立的 emit 函数。

```
微核模板 (唯一的 emit_quant_gemm_inline):
  输入: (QuantFormatDescriptor, DeviceProfile)
  
  Prologue 阶段:
    scale 加载 → 由 ScaleLayout 驱动 (BlockScalar / Hierarchical / ExternalArray)
    zero 加载  → 由 ZeroLayout 驱动 (None / BlockScalar / StaticBias / Hierarchical)
  
  Compute 阶段:
    权重步长     → 由 desc.block_bytes 驱动
    数据展开     → 由 desc.data_kind 驱动 (Int8 直用 / PackedInt4 展开 / PackedInt5/6 展开)
    计算指令     → 由 DeviceProfile.dot_product_cap() 驱动
    累加器类型   → 由 (data_kind, dot_product_cap) 推导: INT32 或 FP32
  
  Epilogue 阶段:
    缩放输出     → int32_to_f32(acc) * scale [+ zero]
```

### §3.2 DeviceProfile 量化能力查询

不按硬件型号 match，而是查询 **能力属性**：

```rust
/// DeviceProfile 提供的量化计算能力查询
impl DeviceProfile {
    /// 最高效的 dot-product 能力
    fn dot_product_cap(&self) -> DotProductCap {
        // 查询顺序: 硬件原生 bf16/fp16 > int4 > int8 > 软件模拟 > 无
        if self.has_bf16_dp()         { DotProductCap::NativeBf16 }
        else if self.has_fp16_dp()    { DotProductCap::NativeFp16 }
        else if self.has_wmma_a4w8()  { DotProductCap::NativeInt4x8 }
        else if self.has_fp4_tc()     { DotProductCap::NativeFp4 }
        else if self.has_int8_tc()    { DotProductCap::NativeInt8Tc }
        else if self.has_avx512_vnni() || self.has_avx_vnni()
                                      { DotProductCap::NativeInt8Simd }
        else if self.has_sve2_sdot()  { DotProductCap::NativeInt8Simd }
        else if self.has_amx()        { DotProductCap::NativeInt8Tile }
        else if self.simd_width() >= 256
                                      { DotProductCap::SimdAssisted }
        else if self.simd_width() >= 128
                                      { DotProductCap::SimdBasic }
        else                          { DotProductCap::None }
    }

    /// 微核 MR×NR 由 SIMD 宽度和寄存器数量自动推导
    fn gemm_tile(&self) -> (usize, usize) {
        // 基于 simd_width / num_regs / accumulator_type 推导
    }
}

#[derive(Debug, Clone, Copy)]
enum DotProductCap {
    // === 原生浮点 dot-product ===
    /// 硬件原生 BF16 dot-product (x86 VDPBF16PS / ARM BFMMLA / GPU HMMA bf16)
    NativeBf16,
    /// 硬件原生 FP16 dot-product (ARM FMMLA / GPU HMMA fp16)
    NativeFp16,

    // === 原生整型 dot-product ===
    /// 硬件原生 4-bit×8-bit (GFX12 a4w8)
    NativeInt4x8,
    /// 硬件原生 FP4 tensor core (SM100+)
    NativeFp4,
    /// 硬件原生 INT8 tensor core (SM80/SM90 IMMA/WGMMA)
    NativeInt8Tc,
    /// 硬件原生 INT8 SIMD dot-product (VNNI VPDPBUSD / SVE2 SDOT)
    NativeInt8Simd,
    /// 硬件原生 INT8 tile (AMX TDPBSSD)
    NativeInt8Tile,

    // === 软件辅助 ===
    /// 有 SIMD 但无 dot-product 指令 (AVX2/AVX-512 FMA, NEON FMLA)
    SimdAssisted,
    /// 基础 SIMD (SSE2, 128-bit only)
    SimdBasic,
    /// 无 SIMD
    None,
}
```

### §3.3 微核三阶段 — 参数化生成

#### Prologue：Scale/Zero 加载

由 `desc.scale_layout` 和 `desc.zero_layout` 驱动，与硬件无关：

```rust
fn emit_quant_prologue(prog, desc, block_ptr, lane_offset) -> (scale_vreg, zero_vreg) {
    // scale — 统一由 ScaleLayout 枚举驱动
    let scale = match &desc.scale_layout {
        BlockScalar { offset, dtype } 
            => emit_load_block_scalar(prog, block_ptr, *offset, dtype),
        BlockScalarWithMin { d_offset, m_offset, dtype }
            => emit_load_block_scalar_min(prog, block_ptr, *d_offset, *m_offset, dtype),
        Hierarchical { block_d_offset, sub_scales_offset, sub_block_elements, .. }
            => emit_load_hierarchical_scale(prog, block_ptr, lane_offset, ..),
        ExternalArray { stride, dtype }
            => emit_load_external_scale(prog, scales_ptr, lane_offset, *stride, dtype),
    };

    // zero — 统一由 ZeroLayout 枚举驱动
    let zero = match &desc.zero_layout {
        ZeroLayout::None => NONE_VREG,
        ZeroLayout::StaticBias { value } => emit_const_bias(prog, *value),
        ZeroLayout::BlockScalar { offset, dtype } 
            => emit_load_block_scalar(prog, block_ptr, *offset, dtype),
        ZeroLayout::Hierarchical { .. } => emit_load_hierarchical_zero(prog, ..),
    };

    (scale, zero)
}
```

#### Compute：数据展开 + Dot-Product

由 `(desc.data_kind, device_profile.dot_product_cap())` 联合驱动：

```rust
fn emit_quant_compute(
    prog, desc, hw, block_ptr, input_ptr, acc, scale, width
) -> new_acc {
    let dot_cap = hw.dot_product_cap();
    let weight_step = desc.block_bytes;
    let blocks_per_tile = k / desc.block_size;

    prog.emit_loop(blocks_per_tile, |prog, b| {
        let block_addr = offset_by(block_ptr, b * weight_step);

        // BF16/FP16/F32: 无 scale/zero，prologue 跳过
        let (scale, zero) = match desc.data_kind {
            Bfloat16 | Float16 | Float32 => (NONE_VREG, NONE_VREG),
            _ => emit_quant_prologue(prog, desc, block_addr, ..),
        };

        // 1. 根据 data_kind 展开量化数据 → 统一的 "element stream"
        let weight_elements = emit_unpack_weight(prog, desc, block_addr, dot_cap);

        // 2. 加载/量化激活
        let activation = emit_load_activation(prog, input_ptr, b, desc.block_size, dot_cap);

        // 3. 根据 dot_cap 选择计算指令 — 这里是多态点
        acc = emit_dot_product(prog, dot_cap, acc, activation, weight_elements, width);

        // 4. Epilogue: 整型路径需要缩放，浮点路径不需要
        match desc.data_kind {
            Bfloat16 | Float16 | Float32 => {},  // 累加已是 FP32，无需缩放
            _ => acc = emit_scale_apply(prog, acc, scale, zero, dot_cap),
        }
    });
    acc
}
```

#### emit_unpack_weight — 按数据类型展开

```rust
/// 将量化权重展开为 dot_product_cap 可以消费的格式
/// 返回值的语义由 dot_cap 和 data_kind 联合决定:
///   NativeBf16/NativeFp16: 返回 BF16/FP16 向量 (直接喂给浮点 dot-product)
///   NativeInt8*: 返回 INT8 向量 (直接喂给 dot-product 指令)
///   NativeInt4x8: 返回 packed INT4 (硬件直接消费)
///   SimdAssisted: 返回 INT8 向量 (软件展开后)
///   None/Basic:   返回 F32 向量 (最终 fallback)
fn emit_unpack_weight(prog, desc, block_addr, dot_cap) -> VRegId {
    let raw = emit_raw_load(prog, block_addr + desc.data_offset, desc.block_bytes - desc.data_offset);

    match desc.data_kind {
        // 原生浮点 — 无需展开，直接返回
        Bfloat16 => raw,   // BF16 packed data，硬件直接消费
        Float16  => raw,   // FP16 packed data，硬件直接消费
        Float32  => raw,   // F32 data，标准 FMA 路径

        Int8 => raw,  // 已是 INT8，无需展开
        
        PackedInt4 | SignedPackedInt4 => {
            match dot_cap {
                NativeInt4x8 => raw,  // 硬件直接消费 packed 4-bit
                _ => {
                    // 软件展开: nibble → sign-extend → INT8
                    let lo = emit_and_mask(prog, raw, 0x0F);
                    let hi = emit_shift_right(prog, raw, 4);
                    let hi = emit_and_mask(prog, hi, 0x0F);
                    let combined = emit_interleave(prog, lo, hi);
                    if desc.data_kind == SignedPackedInt4 {
                        emit_sub_bias(prog, combined, 8)  // Q4_0: 减 8
                    } else {
                        combined
                    }
                }
            }
        }
        
        PackedInt5 => {
            // qs[i] | (qh_bit << 4) → 5-bit → sign-extend → INT8
            let qs = emit_raw_load(prog, block_addr + qs_offset);
            let qh = emit_raw_load(prog, block_addr + qh_offset);
            let merged = emit_bit_merge(prog, qs, qh, 4);
            emit_sub_bias(prog, merged, 16)  // Q5_0: 减 16
        }
        
        PackedInt6 => { /* 类似 PackedInt5，用 2-bit qh */ }
        Float4 => {
            match dot_cap {
                NativeFp4 => raw,  // 硬件直接消费 FP4
                _ => emit_fp4_decode(prog, raw),  // e2m1 → F32
            }
        }
        SuperLowBit => emit_codebook_lookup(prog, raw, desc.codebook),
    }
}
```

#### emit_dot_product — 多态计算核心

**这是唯一需要感知硬件指令的地方**，但通过 `dot_cap` 枚举驱动而非硬件型号：

```rust
/// 根据 dot_product_cap 选择硬件原生 dot-product 指令
/// 调用方不关心具体硬件，只关心"累加器类型"和"吞吐"
fn emit_dot_product(prog, dot_cap, acc, a, b, width) -> VRegId {
    match dot_cap {
        // 原生浮点 dot-product — 累加在 FP32
        NativeBf16 => prog.emit(Bf16Dot { acc, a, b, width }),     // VDPBF16PS / BFMMLA / HMMA bf16
        NativeFp16 => prog.emit(Fp16Dot { acc, a, b, width }),     // FMMLA / HMMA fp16

        // 硬件 INT8 dot-product — 累加在 INT32
        NativeInt4x8   => prog.emit(WmmaA4W8 { acc, a, b }),
        NativeInt8Tc   => prog.emit(GpuInt8Dot { acc, a, b, width }),
        NativeInt8Simd => prog.emit(SimdInt8Dot { acc, a, b, width }),  // VPDPBUSD / SDOT
        NativeInt8Tile => prog.emit(TileInt8Dot { acc, a, b }),         // AMX TDPBSSD

        // 有 SIMD 但无 dot-product — INT8 手动乘加
        SimdAssisted => {
            let products = prog.emit(Int8Mul { a, b, width });
            prog.emit(Int32Accumulate { acc, products })
        }

        // 基础 SIMD — F32 FMA (INT8 展开 → F32 后)
        SimdBasic => prog.emit(Fma { acc, a, b }),

        // 无 SIMD — 标量 (仍然是 JIT 生成的)
        None => prog.emit(ScalarFma { acc, a, b }),
    }
}
```

**关键设计**：
- `emit_dot_product` 只 match `DotProductCap` 枚举（10 个变体），不 match 硬件型号
- BF16/FP16 路径累加在 FP32（VDPBF16PS 输出 f32），整型路径累加在 INT32
- ISA lowering 层把 `Bf16Dot` / `Fp16Dot` / `SimdInt8Dot` 映射到具体指令（x86→VDPBF16PS/FMMLA/VPDPBUSD, ARM→BFMMLA/FMMLA/SDOT），codegen 不感知
- 新增硬件只需在 `DeviceProfile.dot_product_cap()` 中加一条检测，微核模板自动适配

### §3.4 完整的 QuantGemm 入口

```rust
/// 统一 GEMM 入口 — 浮点与量化共享同一微核模板
/// quant_type = Some(Bf16) → BF16 原生 GEMM (VDPBF16PS/BFMMLA/HMMA)
/// quant_type = Some(F16)  → FP16 原生 GEMM (FMMLA/HMMA fp16)
/// quant_type = Some(F32)  → 标准 FMA GEMM
/// quant_type = Some(Q4_0) → INT4 参数化量化微核
/// quant_type = None        → 从 graph dtype 推导 (BF16/F32)
fn emit_gemm_inline(
    prog: &mut VmProgram,
    m: BoundExpr, n: usize, k: usize,
    quant_type: Option<QuantType>,
    width: SimdWidth,
    input_ptr: VRegId, weight_ptr: VRegId, output_ptr: VRegId,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    let qt = quant_type.unwrap_or_else(|| match dtype {
        QuantPrecision::BF16 => QuantType::Bf16,
        _ => QuantType::F32,
    });
    let desc = QUANT_REGISTRY.get(qt);
    let hw = prog.device_profile();
    let dot_cap = hw.dot_product_cap();
    let (mr, nr) = hw.gemm_tile();

    // 统一的参数化微核 — 所有格式走同一路径
    emit_quant_gemm_tiled(prog, desc, dot_cap, m, n, k,
                          mr, nr, width, input_ptr, weight_ptr, output_ptr)
}
```

**不存在** `emit_vnni_int8_gemm`、`emit_sm90_int8_gemm` 等按硬件命名的函数。
**只有** `emit_quant_gemm_tiled` — 一个参数化模板，`dot_cap` 和 `desc` 决定一切。

### §3.5 其他量化算子

#### §3.5.1 QuantGather — 量化 Embedding 查表

Gather 不做 dot-product，语义是"从量化表中读出行并还原为 compute_dtype hidden state"。compute_dtype 由 TensorMeta.dtype × DeviceProfile 推导，量化路径通常为 F32 累加后输出，原生浮点路径保持原始精度：

```rust
fn emit_quant_gather_inline(prog, desc, dot_cap, ...) {
    prog.emit_loop(seq_len, |prog, token_idx| {
        let token_id = load_token_id(prog, indices_ptr, token_idx);
        let row_addr = compute_row_addr(embed_ptr, token_id, desc);

        prog.emit_loop(hidden_dim / desc.block_size, |prog, b| {
            let block_addr = offset_by(row_addr, b * desc.block_bytes);
            let (scale, zero) = emit_quant_prologue(prog, desc, block_addr, ..);
            
            // 展开 + 缩放 → 直接写出 compute_dtype
            let elements = emit_unpack_weight(prog, desc, block_addr, dot_cap);
            let scaled_values = emit_scale_output(prog, elements, scale, zero, dot_cap, ctx.dtype);
            store_output(prog, output_ptr, token_idx, b, scaled_values);
        });
    });
}
```

#### §3.5.2 QuantAttention — 量化 KV Cache

KV cache 量化时，attention 的 K/V 读取复用 `emit_unpack_weight` + `emit_dot_product`：

```
Q·K^T = dot_product(Q_quant, K_quant, dot_cap) × scale_q × scale_k
softmax(Q·K^T) · V = dot_product(softmax_quant, V_quant, dot_cap) × scale_v
```

#### §3.5.3 QuantMoE — 量化 Expert FFN

每个 expert 的 GEMM 复用 `emit_gemm_inline(prog, ..., quant_type=Some(expert_qt), ...)`。同一微核模板，不同 expert 可以有不同的 `quant_type`。

## §4 VmInstr 设计原则

### §4.1 硬件无关的 VM 指令

新增的量化 VmInstr 是**语义操作**，不编码硬件细节：

```rust
// === 数据展开 (语义操作) ===
QuantWeightLoad { dst, base, offset, bytes }     // 加载 raw quant bytes
NibbleUnpack { dst, src }                        // packed nibble → 2× INT8 elements
SignExtend { dst, src, bias }                    // 减偏移 (参数化: 8/16/32)
HighBitMerge { dst, low, high, shift }           // qs | (qh << shift)
CodebookLookup { dst, indices, table, size }     // 索引 → codebook 值

// === Scale 操作 ===
ScaleLoad { dst, base, offset, dtype }           // 加载 scale (f16/bf16/f32 → broadcast)
SubScaleLoad { dst, base, block_idx, sub_idx }   // K-Quant 子块 scale
ScaleCombine { dst, block_d, sub_scale }          // block_d * sub_scale
ScaleApplyInt { dst, int_acc, scale, zero }       // int32 → f32 * scale + zero
ScaleApplyFp { dst, fp_acc, scale, zero }         // fp32 * scale + zero

// === Dot-Product (语义操作 → ISA lowering 决定具体指令) ===
Int8Dot { acc, a, b }                            // int32 += int8 · int8 (硬件无关)
Int4x8Dot { acc, a_4bit, b_8bit }                // int32 += int4 · int8 (硬件无关)
Fp4Dot { acc, a, b }                             // fp32 += fp4 · fp4 (硬件无关)
```

### §4.2 ISA Lowering 的多态映射

VmInstr 是硬件无关的语义操作，**一条 `Int8Dot` 在不同 ISA 上 lower 为不同指令**：

| VmInstr | x86_64 (VNNI+BF16) | x86_64 (AMX) | AArch64 (SVE2/SME2) | GPU (SM80) | GPU (SM90) | GPU (GFX12) |
|---------|--------------------|--------------|---------------------|------------|------------|-------------|
| `Bf16Dot` | `VDPBF16PS` | `TDPBF16PS` | `BFMMLA` | `HMMA.bf16` | `WGMMA.bf16` | `WMMA.bf16` |
| `Fp16Dot` | N/A (x86 无 FP16 计算) | N/A | `FMMLA` | `HMMA.fp16` | `WGMMA.fp16` | `WMMA.fp16` |
| `Int8Dot` | `vpdpbusd` | `TDPBSSD` | `SDOT` | `m16n8k32.s32.u8.u8` | `wgmma.int8` | `v_wmma_i32_iu8` |
| `Int4x8Dot` | 软件展开+`vpdpbusd` | 软件展开+`TDPBSSD` | 软件展开+`SDOT` | 软件展开+IMMA | 软件展开+`wgmma` | `v_wmma_i32_iu4_iu8` 原生 |
| `Fp4Dot` | N/A | N/A | N/A | N/A | N/A (SM90) / `tcgen05.fp4` (SM100+) | N/A |

**新增硬件平台 = 新增一行 ISA lowering 映射，微核模板和 VmInstr 定义不变。**

### §4.3 新增 VmInstr 完整清单

```rust
// === 权重加载 ===
QuantWeightLoad { dst, base, offset, bytes }     // raw quant bytes → VReg

// === 数据展开 ===
NibbleUnpack { dst, src }                        // packed nibble → 2x elements
SignExtend { dst, src, bias }                    // sub bias (parametric)
HighBitMerge { dst, low, high, shift }           // bit-merge for 5/6 bit
CodebookLookup { dst, indices, table_ptr, table_len }  // IQ codebook

// === Scale ===
ScaleLoad { dst, base, offset, dtype }           // f16/bf16/f32 → broadcast
SubScaleLoad { dst, base, block_idx, sub_idx }   // K-Quant sub-scale
ScaleCombine { dst, block_d, sub_scale }          // d * sub_scale
ScaleApplyInt { dst, int_acc, scale, zero }       // int32→f32, *scale, +zero
ScaleApplyFp { dst, fp_acc, scale, zero }         // fp32 *scale +zero

// === Dot-Product (hardware-agnostic semantics) ===
Int8Dot { acc, a, b }                            // int8·int8→int32 accumulate
Int4x8Dot { acc, a_4bit, b_8bit }                // int4·int8→int32 accumulate
Fp4Dot { acc, a, b }                             // fp4·fp4→fp32 accumulate
Bf16Dot { acc, a, b, width }                     // bf16·bf16→fp32 accumulate (VDPBF16PS/BFMMLA/HMMA)
Fp16Dot { acc, a, b, width }                     // fp16·fp16→fp32 accumulate (FMMLA/HMMA fp16)

// === Activation 量化 (input f32 → int8 for quant dot-product) ===
QuantizeF32ToI8 { dst, src, scale }              // f32 → i8 (for quant gemm input)

// === DequantFMA Fallback (量化模型最终兜底) ===
QuantBlockDequant { dst, base, offset, desc_idx, dtype } // 整块解码 → compute_dtype 向量
```

## §5 量化感知融合

### §5.1 融合规则

**同精度算子自由融合**（零开销）：

```
QuantGemm(Q4_0) + ResidualAdd(F32) + RmsNorm(F32)
  → GEMM 输出 F32 (epilogue: int32*scale→f32)
  → 后续 F32 算子正常融合
  → 全链路: QuantGemm prologue/compute/epilogue + Residual + Norm 一气呵成
```

**不同精度算子必须拆分**：

```
QuantGemm(Q4_0) → QuantGemm(Q6K)  // 输出 F32 → 输入需要 packed Q6K? 不合理
// GEMM 之间不会直接相连（中间有激活函数/Norm），所以这不是常见场景
```

### §5.2 混合精度模型的处理

一个模型可能同时使用多种量化格式（如 Q4_0 大部分层 + Q6K lm_head）：

```
每层编译为独立的融合图:
  Layer 0..N-1 (Q4_0): FusedAttentionLayer_Q4_0 + FusedFfnLayer_Q4_0
  Layer N (lm_head, Q6K): FusedFfnLayer_Q6K

每层编译时传入该层的 quant_type → 对应路径的机器码
不同层的机器码可以完全不同（不同路径、不同指令）
```

## §6 REQ 清单

### REQ-QCG-001: QuantFormatDescriptor 元数据
- 实现 `QuantFormatDescriptor` 含 `data_kind: QuantDataKind` 字段
- 注册 §0 清单中全部 QuantType（按族注册，同族共享 data_kind）
- 验证：每种 QuantType 能查到 descriptor，`data_kind` 正确分类

### REQ-QCG-002: VmInstr 原生量化指令集
- 新增 §4.3 列出的全部 VmInstr 变体
- 每种指令在 x86_64 ISA lowering 实现
- 每种指令在 reg_alloc/verify 中正确处理
- 验证：单元测试每种指令的 lowering

### REQ-QCG-003: 参数化微核模板
- 实现 `emit_quant_gemm_tiled` 统一微核模板
- 实现 `emit_quant_prologue` / `emit_unpack_weight` / `emit_dot_product` 参数化组件
- 实现 `DeviceProfile.dot_product_cap()` 能力查询
- 验证：同一模板在不同 DeviceProfile 下生成不同指令

### REQ-QCG-004: INT8 数据路径 (Q8_0/Q8_1/Q8K)
- `Int8Dot` VmInstr 的各 ISA lowering (VPDPBUSD / SDOT / IMMA / WMMA)
- `QuantizeF32ToI8` 激活量化指令
- E2E 测试: Q8_0 GGUF 模型推理

### REQ-QCG-005: INT4 数据路径 (Q4_0/Q4_1/Q4K)
- `NibbleUnpack` + `SignExtend` + `Int8Dot` 组合路径
- `Int4x8Dot` 原生路径 (GFX12 a4w8 ISA lowering)
- E2E 测试: Q4_0/Q4_1 GGUF 模型推理

### REQ-QCG-006: INT5/INT6 数据路径 (Q5_0/Q5_1/Q5K/Q6K)
- `HighBitMerge` 指令 (qs+qh 合并)
- 5/6-bit 展开 → INT8 dot-product 路径
- E2E 测试: Q5_0/Q5K/Q6K 模型推理

### REQ-QCG-007: 极低比特路径 (Q2K/Q3K)
- 2-bit/3-bit 特殊展开 + `SubScaleLoad` / `ScaleCombine`
- K-Quant 多级 scale 加载
- E2E 测试: Q2K/Q3K 模型推理

### REQ-QCG-008: IQ Codebook 路径 (IQ1S..IQ4XS)
- `CodebookLookup` VmInstr 实现
- Codebook 常量数据嵌入 JIT 机器码 (.rodata / .const)
- IQ 系列数学语义决定其输出是 F32（codebook 查表结果是 float）
- E2E 测试: IQ4_NL 模型推理

### REQ-QCG-009: FP4 数据路径 (MXFP4/NVFP4)
- `Fp4Dot` VmInstr 及 ISA lowering
- SM100+: tcgen05 FP4 原生路径 (Nvfp4 data_kind)
- SM<100 / CPU: e2m1 软件查表解码 → F32 → FMA (MXFP4 Float4 data_kind)
- NVFP4 两级缩放: `value = global_f32 × UE4M3_scale[sub_block] × e2m1_lut[qs]`
- MXFP4 单级缩放: `value = E8M0_scale[block] × e2m1_lut[qs]`，E8M0 = `2^(byte-127)`
- NVFP4 与 MXFP4 共享 E2M1 16 值真值表，差异仅在缩放模型
- E2E 测试: MXFP4 / NVFP4 模型推理

### REQ-QCG-009a: NVFP4 E2M1 硬件原生路径
- SM100+ `Nvfp4` data_kind 走 `emit_quant_gemm_nvfp4`，使用 Nvfp4SubBlockDequant + Fp4Dot
- sub-block 16 元素 × 4 个 sub-block per block
- 4 × UE4M3 FP8 缩放融合进 Epilogue
- x86/AArch64 无 FP4 硬件，走 MXFP4 共享的 E2M1 查表路径

### REQ-QCG-010: AWQ4 Weight-Only 反量化路径
- AWQ4 是 Weight-Only 整数量化: 激活保持 FP16，权重压缩到 4-bit
- 反量化公式: `FP16_weight = (qw - zero_point_FP16) × scale_FP16`
- 行优先连续 nibble 打包 (8 个连续权重 per u32)
- GEMM 路径: 加载 u32 → 位移+掩码解包 4-bit → 减 zero_point → 乘 scale → FP16 HMMA
- 反量化在寄存器中完成，不写回显存
- GGUF 原生支持: GGUF 文件可直接包含 AWQ4 量化权重，gllm 加载后直接走原生 AWQ4 反量化路径
- HuggingFace safetensors 直加载也走同一原生路径
- 验证: AWQ4 格式权重的 GEMM 输出与 FP16 参考 bit-exact (在量化精度范围内)

### REQ-QCG-010a: GPTQ4 Weight-Only 反量化路径
- GPTQ4 同属 Weight-Only 整数量化，反量化公式与 AWQ4 相同
- 关键差异: 列交织打包 (stride-16)，zero-point 为 INT4 packed (8 个 zp 打包进 u32，需 +1 偏移)
- 列交织使 Warp 32 线程同时访问同一 u32 的不同 nibble，避免银行冲突
- 解包: `zp_i = (zeros >> (i * 4)) & 0xF + 1`
- GEMM 路径: 与 AWQ4 相同的寄存器内反量化 → FP16 HMMA
- GGUF 原生支持: 同 AWQ4，GGUF 文件可直接包含 GPTQ4 权重
- 验证: GPTQ4 格式权重的 GEMM 输出与 FP16 参考 bit-exact

### REQ-QCG-013: QuantGather 算子
- 量化 embedding 查表: 逐 block 解码 → 直接写出 compute_dtype hidden state (dtype 由 TensorMeta 推导)
- 移除 Rust 端所有反量化代码
- E2E 测试: 量化 embed 模型的 token embedding 正确

### REQ-QCG-011: 量化感知融合
- `can_fuse_quant_aware` 融合判断
- 同精度算子吸附，不同精度拆分
- 混合精度模型按层独立编译

### REQ-QCG-012: 移除 Rust 端反量化
- `grep -rn "dequant" gllm/src/engine/` 无匹配
- 所有量化权重原格式直传 JIT
- E2E: 混合量化 GGUF (Q4_0 + Q6K lm_head) 直接推理

## §7 OpKind 修改

### §7.1 QuantGemm OpKind 携带 QuantType

```rust
// 修改前:
QuantGemm { m: SymDim, n: usize, k: usize, block_size: usize, bits: usize, scale_offset: usize, zero_point_offset: Option<usize> }

// 修改后:
QuantGemm { m: SymDim, n: usize, k: usize, quant_type: QuantType }
```

`QuantType` 已有 `block_size()` / `bits()` / `block_bytes()` 方法，分解参数不再需要。

### §7.2 新增 QuantGather OpKind

```rust
QuantGather {
    indices_sym: String,     // token ID 输入
    table_size: usize,       // vocab_size
    hidden_dim: usize,       // output dim
    quant_type: QuantType,   // embed 权重的量化格式
}
```

## §8 实施顺序

```
Phase 1: 参数化微核基础设施
  Step 1: QuantFormatDescriptor + QuantDataKind (REQ-QCG-001)
  Step 2: VmInstr 原生量化指令集 — 硬件无关语义 (REQ-QCG-002)
  Step 3: DeviceProfile.dot_product_cap() + gemm_tile() (REQ-QCG-003 部分)
  Step 4: QuantGemm OpKind 改为携带 QuantType (§7.1)
  Step 5: emit_quant_gemm_tiled 参数化微核模板 (REQ-QCG-003)
    — emit_quant_prologue / emit_unpack_weight / emit_dot_product 三组件

Phase 2: INT8 路径 (Q8_0/Q8_1/Q8K)
  Step 6: Int8Dot ISA lowering (x86 VNNI / ARM SDOT / GPU IMMA) (REQ-QCG-004)
  Step 7: INT8 E2E 测试 (Q8_0 模型)

Phase 3: INT4 路径 (Q4_0/Q4_1/Q4K)
  Step 8: NibbleUnpack + SignExtend + Int4x8Dot ISA lowering (REQ-QCG-005)
  Step 9: INT4 E2E 测试 (Q4_0/Q4_1 模型)

Phase 4: INT5/INT6 路径 (Q5_0/Q5_1/Q5K/Q6K)
  Step 10: HighBitMerge + 5/6-bit 展开组件 (REQ-QCG-006)
  Step 11: INT5/INT6 E2E 测试

Phase 5: 极低比特 + IQ + FP4
  Step 12: Q2K/Q3K + K-Quant 多级 scale (REQ-QCG-007)
  Step 13: IQ codebook (REQ-QCG-008)
  Step 14: FP4 (REQ-QCG-009)

Phase 6: 集成
  Step 15: QuantGather 算子 (REQ-QCG-010)
  Step 16: 量化感知融合 (REQ-QCG-011)
  Step 17: 移除 Rust 反量化 + 全格式 E2E 矩阵 (REQ-QCG-012)
```

## §9 与其他 SPEC 的关系

| 章节 | 关系 | 说明 |
|------|------|------|
| §01-JIT-PIPELINE.md §5.5 DequantComputeVariant | **被本文件取代** | 原表降为 §4 路径选择器的输入 |
| §01-JIT-PIPELINE.md §5.1 自动指令选择 | 复用 | 量化 VmInstr 走标准 auto_lower_trace |
| §14-HW-INTRINSICS.md | **输入** | 硬件指令能力矩阵，路径选择器的硬件检测依据 |
| §16-DEVICE-FUSION.md REQ-FUS-008 量化感知融合 | **扩展** | 本文件 §5 提供具体融合规则 |
| §22-PAGE-COMPRESSION.md | 正交 | §22 是页字节流 codec，本文件是算子内原生量化计算 |
| §07-LOADER.md ARCH-MXFP4-SEPARATE | 兼容 | MXFP4 的 ExternalArray scale 在 QuantFormatDescriptor 中表达 |
