# NVFP 技术参考 (NVIDIA Microscaling FP4/FP8)

> **用途**: 本文档为 gllm NVFP 量化格式家族实现提供底层技术参考。
> 涵盖 NVFP 家族全景 (NVFP4 + NVFP8)、与传统量化格式硬件实现对比、E2M1 编码、UE4M3 子块缩放、硬件 bit-packing 规约、Scale 对齐约束、CUDA intrinsics、PTX 指令等。
> 实现规范见 `SPEC/23-QUANT-CODEGEN-ALGO.md`。

## §1 NVFP 家族全景

### §1.1 定义

NVFP 是 NVIDIA 专门为 Blackwell 架构 (SM100) 定制的、绑定微块缩放 (Microscaling) 技术的专用浮点量化格式群。
核心二进制结构只有 4-bit 和 8-bit 两种 — 因为 4 位 (1 Nibble/半字节) 和 8 位 (1 Byte) 可以完美对齐到 32 位或 64 位寄存器, 而 3 位、6 位会导致严重的硬件移位对齐开销。

| 格式 | 核心二进制 | 微块缩放 | 硬件支持 | 引擎实操意义 |
|------|-----------|---------|---------|------------|
| **NVFP4** | E2M1 (4-bit) | 每 16 元素共享 1 个 UE4M3 Scale + 全局 F32 Scale | Blackwell (SM100) 独占硬件原生加速 | 吞吐量 FP16 的 4x, 大模型权重极限压缩首选 |
| **NVFP8** | E4M3 或 E5M2 (8-bit) | 每 32 元素共享 1 个 Scale (OCP MX 开放标准变体) | Hopper (SM90) 开始支持, Blackwell 完美优化 | 吞吐量 FP16 的 2x, 计算密集型 Activation 或部分权重 |

### §1.2 NVFP4 与传统 4-bit 量化在机器码层面的本质区别

这是推理引擎开发者最需要理解的关键差异:

#### 传统 4-bit 量化 (INT4 / AWQ / GPTQ / GGML Q4_*) — 软件反量化

GPU 没有 `MMA.INT4` 浮点计算核心。机器码指令流:

```
1. LDG — 把打包 INT4 权重从显存加载到寄存器
2. SHL/SHR + AND — 在寄存器中拆解 4-bit nibble
3. 多条算术指令 — 乘以 Scale、减去 Zero-Point → 强行转换为 FP16/BF16
4. mma.sync (FP16) — 最终发射 FP16 矩阵乘法
```

缺点: 大量发射时钟周期 (Issue Cycles) + 高寄存器压力 (Register Pressure)。

#### NVIDIA NVFP4 — 硬件原生硬解 (Blackwell 独占)

Blackwell Tensor Core 内部集成了硬件级 E2M1 解码器:

```
1. 直接发射 mma.sync.aligned.m16n8k32 (FP4 操作数)
   - srcA 寄存器: 存放未经拆解的连续打包 4-bit 原始流
   - srcB 寄存器: 按硬件要求的布局存放 FP8 Block Scale
2. Tensor Core 在计算流水线上同步完成乘加与反量化
3. 零软件位移指令
```

## §2 E2M1 浮点编码 (NVFP4/MXFP4 共用)

### §2.1 位布局

```
FP4 E2M1 (4 bits):

┌───┬───────┬───────┐
│ S │ E[1:0]│  M    │
├───┼───────┼───────┤
│ 3 │  2–1  │   0   │
└───┴───────┴───────┘

S = Sign bit (1 bit)
E = Exponent (2 bits)
M = Mantissa (1 bit)
```

### §2.2 解码规则

- **Normal** (E != 0): `value = (-1)^S × 2^(E-1) × (1 + M×0.5)`
- **Subnormal** (E == 0): `value = (-1)^S × 2^(-2) × (M×0.5)`

### §2.3 完整真值表 (16 个值)

| Index | [S,E,M] | 计算 | 值 |
|-------|---------|------|----|
| 0b0000 | [0,00,0] | +2^(-2)×0 | **0** |
| 0b0001 | [0,00,1] | +2^(-2)×0.5 | **+0.5** |
| 0b0010 | [0,01,0] | +2^0×1 | **+1** |
| 0b0011 | [0,01,1] | +2^0×1.5 | **+1.5** |
| 0b0100 | [0,10,0] | +2^1×1 | **+2** |
| 0b0101 | [0,10,1] | +2^1×1.5 | **+3** |
| 0b0110 | [0,11,0] | +2^2×1 | **+4** |
| 0b0111 | [0,11,1] | +2^2×1.5 | **+6** |
| 0b1000 | [1,00,0] | -2^(-2)×0 | **-0** |
| 0b1001 | [1,00,1] | -2^(-2)×0.5 | **-0.5** |
| 0b1010 | [1,01,0] | -2^0×1 | **-1** |
| 0b1011 | [1,01,1] | -2^0×1.5 | **-1.5** |
| 0b1100 | [1,10,0] | -2^1×1 | **-2** |
| 0b1101 | [1,10,1] | -2^1×1.5 | **-3** |
| 0b1110 | [1,11,0] | -2^2×1 | **-4** |
| 0b1111 | [1,11,1] | -2^2×1.5 | **-6** |

**LUT (无符号索引)**: `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]`

与 MXFP4 共用同一 LUT (`kvalues_mxfp4` / `ggml_table_f32_e4`), 参见 llama.cpp。

## §3 NVFP8 浮点编码 (E4M3 / E5M2)

### §3.1 概述

NVFP8 使用 OCP (Open Compute Project) 标准的 FP8 编码, 与 NVFP4 的 E2M1 不同:

| 变体 | 符号 | 指数 | 尾数 | Bias | 动态范围 | 典型用途 |
|------|------|------|------|------|---------|---------|
| **E4M3** | 1 bit | 4 bits | 3 bits | 7 | ±448 | 前向权重 + Activation (更高精度) |
| **E5M2** | 1 bit | 5 bits | 2 bits | 15 | ±57344 | 反向梯度 (更大动态范围) |

### §3.2 E4M3 位布局

```
FP8 E4M3 (8 bits):

┌───┬───────────┬───────────┐
│ S │  E[3:0]   │   M[2:0]  │
├───┼───────────┼───────────┤
│ 7 │   6–3     │    2–0    │
└───┴───────────┴───────────┘

Bias = 7
Normal: (-1)^S × 2^(E-7) × (1 + M/8)   (E: 1..14, 不含 15)
Subnormal: (-1)^S × 2^(-6) × (M/8)      (E = 0)
NaN: E=15, M != 0
Inf: 不存在 (E4M3 没有 Inf 编码)
```

### §3.3 E5M2 位布局

```
FP8 E5M2 (8 bits):

┌───┬───────────┬───────────┐
│ S │  E[4:0]   │   M[1:0]  │
├───┼───────────┼───────────┤
│ 7 │   6–2     │    1–0    │
└───┴───────────┴───────────┘

Bias = 15
Normal: (-1)^S × 2^(E-15) × (1 + M/4)   (E: 1..30)
Subnormal: (-1)^S × 2^(-14) × (M/4)      (E = 0)
Inf: E=31, M=0
NaN: E=31, M != 0
```

### §3.4 NVFP8 与 NVFP4 的关系

NVFP8 不是 NVFP4 的简单扩展。两者在硬件集成中的角色不同:
- **NVFP8**: Hopper (SM90) 起支持, 主要用于 Activation 和混合精度计算
- **NVFP4**: Blackwell (SM100) 独占, 主要用于权重极限压缩
- 两者共享 Microscaling 思想 (微块共享缩放因子), 但缩放粒度和格式不同

## §4 两级缩放模型 (NVFP4 专用)

### §4.1 公式

```
Final_value_FP32 = E2M1_value(nibble) × UE4M3_scale(sub_block_idx) × F32_global_tensor_scale
```

- **E2M1_value**: §2.3 真值表中的值 (4-bit lookup)
- **UE4M3_scale**: 子块级缩放因子 (8-bit unsigned FP8 E4M3, 每子块一个)
- **F32_global_tensor_scale**: 全局张量级缩放因子 (FP32, 独立于 block 存储)

### §4.2 与 MXFP4 对比

| 特性 | NVFP4 | MXFP4 |
|------|-------|-------|
| 子块大小 | 16 elements | 32 elements (= block) |
| 子块缩放 | UE4M3 (FP8 E4M3, 无符号) | E8M0 (power-of-2 only) |
| 缩放范围 | ~2^(-6) 到 ~2^(8.875) | 仅 2 的整数次幂 |
| 全局缩放 | 有 (F32 tensor scale) | 无 |
| 动态范围 | 更宽 (UE4M3 非线性) | 更窄 (纯幂次) |
| GGML type | 40 | 39 |
| Block size | 64 elements / 36 bytes | 32 elements / 17 bytes |

## §5 UE4M3 (Unsigned FP8 E4M3) 编码

NVFP4 子块缩放使用 UE4M3 格式 (无符号 FP8):

```
┌───────────┬───────────────────┐
│ E[3:0]    │      M[2:0]       │
├───────────┼───────────────────┤
│  7–4      │       3–1         │
└───────────┴───────────────────┘

Bias = 7
4 exponent bits, 3 mantissa bits
No sign bit (unsigned)
```

### §5.1 解码规则

- **Normal** (1 ≤ E ≤ 15): `scale = 2^(E-7) × (1 + M/8)`
- **Subnormal** (E = 0): `scale = 2^(-6) × (M/8)`
- **范围**: 0 (when M=0,E=0) ~ 2^(8.875) ≈ 480.0

### §5.2 软件解码伪码

```c
float ue4m3_to_f32(uint8_t raw) {
    uint32_t exp = (raw >> 3) & 0xF;
    uint32_t mant = raw & 0x7;
    if (exp == 0) {
        // subnormal: 2^(-6) * (mant / 8)
        return (float)mant / 8.0f * 0.015625f;
    }
    // normal: 2^(exp-7) * (1 + mant/8)
    float sign_val = 1.0f + (float)mant / 8.0f;
    return sign_val * exp2f((float)exp - 7.0f);
}
```

## §6 硬件 Bit-Packing 规约 (NVFP4)

### §6.1 小端序 Nibble 填充

NVFP4 采用小端序 (Little-Endian) 逐个 Nibble 填充。一个 32 位寄存器 (uint32_t) 正好容纳 8 个 NVFP4 元素:

```
32 位寄存器 / 内存单元:

+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| Element 7 | Element 6 | Element 5 | Element 4 | Element 3 | Element 2 | Element 1 | Element 0 |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
[31      28] [27      24] [23      20] [19      16] [15      12] [11       8] [7        4] [3        0]

每个 Element 内部 4-bit 结构:
  Bit 3: Sign
  Bit 2-1: Exponent
  Bit 0: Mantissa
```

### §6.2 Byte 级视图

```
Byte [n]:
  Low nibble (bits 3:0)  = 偶数索引元素 (element 2n)
  High nibble (bits 7:4) = 奇数索引元素 (element 2n+1)
```

与 GGML/llama.cpp 的 `block_nvfp4.qs` 布局完全一致。

### §6.3 ⚠️ 微块缩放因子对齐约束 (最致命的陷阱)

当发射 Blackwell FP4 MMA 指令时, 硬件在 **K 维度** 上要求:

**每 16 个连续的 K 维元素 (共 64 bits = 8 字节) 必须共享一个 FP8 缩放因子 (E4M3 格式)。**

关键约束:
1. **禁止 Scale 与 4-bit 交织存储** — 交织布局会导致硬件寻址极其低效
2. **必须采用 Split-Plane (分离平面) 或 Structure of Arrays (SoA) 布局**
3. **Weight Loading 阶段需要开辟两块独立指针**:
   - `uint32_t* weights` — 纯权重矩阵 (打包 FP4 数据)
   - `__nv_fp8_e4m3* scales` — Scale 矩阵 (每 16 元素一个 FP8)

### §6.4 GGML/llama.cpp 的 Split-Plane 布局

```
block_nvfp4 (64 elements, 36 bytes):

Offset  Size   Content
0       4B     d[4] — 4 个 UE4M3 子块缩放 (Split-Plane: Scale 区)
               d[0] = sub-block 0 (elements 0-15)   ← K 维度每 16 元素
               d[1] = sub-block 1 (elements 16-31)
               d[2] = sub-block 2 (elements 32-47)
               d[3] = sub-block 3 (elements 48-63)
4       32B    qs[32] — 32 字节打包 E2M1 (Split-Plane: Data 区)
               byte[0].lo = element 0, byte[0].hi = element 1
               ...
               byte[31].lo = element 62, byte[31].hi = element 63
```

Scale (4B) 与 Data (32B) 分离存储, 不交织。

## §7 内存布局

### §7.1 Rust 结构体 (`gllm-kernels`)

```rust
const QK_NVFP4: usize = 64;
const QK_NVFP4_SUB: usize = 16;

#[repr(C)]
pub struct BlockNvfp4 {
    pub d: [u8; QK_NVFP4 / QK_NVFP4_SUB],  // 4 UE4M3 scales (Split-Plane)
    pub qs: [u8; QK_NVFP4 / 2],             // 32 packed E2M1 (Split-Plane)
}
// assert!(size_of::<BlockNvfp4>() == 36)
```

### §7.2 全局张量缩放

`F32_global_tensor_scale` 存储在 GGUF 张量元数据中, 独立于 block 数据:
- GGUF key: `tensor_name.scale` 或模型级 metadata
- gllm-kernels 传入方式: 作为 GEMM 的 A 矩阵全局缩放 (arg parameter)

### §7.3 Weight Loading 阶段的指针布局

```c
// 推理引擎加载 NVFP4 权重时, 需要为 MMA 指令准备两块内存:
uint32_t* weight_ptr;        // 纯 FP4 打包权重 (每 uint32 = 8 elements)
__nv_fp8_e4m3* scale_ptr;    // 子块缩放矩阵 (每 16 elements 一个 FP8)

// MMA 指令发射时:
// srcA = weight_ptr + tile_offset
// srcB = scale_ptr + (tile_k / 16)   // K 维度每 16 元素取一个 scale
```

## §8 CUDA Intrinsics (SM100 Blackwell)

### §8.1 头文件

```c
#include <cuda_fp4.h>    // SM100+ Blackwell
```

### §8.2 数据类型

| 类型 | 大小 | 说明 |
|------|------|------|
| `__nv_fp4_e2m1` | 1 byte (2 elements) | 2 个 E2M1 值打包 |
| `__nv_fp4x2_e2m1` | 1 word (4 elements) | 4 个 E2M1 值打包 |
| `__nv_fp4x4_e2m1` | 1 dword (8 elements) | 8 个 E2M1 值打包 |

### §8.3 缩放类型

| 类型 | 大小 | 说明 |
|------|------|------|
| `__nv_fp8_e4m3` | 1 byte | FP8 E4M3 (有符号) |
| UE4M3 (unsigned) | 1 byte | 子块缩放用; 无符号变体, 通过 bit reinterpret |

### §8.4 转换函数

```c
// FP32 → FP4
__nv_fp4_e2m1 __nv_float_to_fp4_e2m1(float val, __nv_saturation_t saturate);
__nv_fp4x2_e2m1 __nv_floatx2_to_fp4x2_e2m1(float2 val, __nv_saturation_t saturate);

// FP4 → FP32
float __nv_fp4_e2m1_to_float(__nv_fp4_e2m1 val);
float2 __nv_fp4x2_e2m1_to_float2(__nv_fp4x2_e2m1 val);
```

## §9 PTX 指令 (SM100 Tensor Core FP4)

### §9.1 MMA 指令

```
mma.sync.aligned.m16n8k32.row.col.f32.f4.f4.f32
    { %d0, %d1, %d2, %d3 },
    %a, %b,
    { %c0, %c1, %c2, %c3 };
```

- **m16n8k32**: M=16, N=8, K=32 (FP4 每 element 4-bit, 32×4/8 = 16 bytes per row)
- **.f4**: FP4 E2M1 operand type
- **accumulator**: FP32
- **throughput**: 1 MMA per clock per TC (SM100)
- **Scale 对齐**: K=32 = 2 个 sub-block, 硬件自动从 scale 矩阵取 2 个 FP8 缩放因子

### §9.2 WGMMA (Warp-level MMA)

```
wgmma.mma_async.sync.aligned.m64nNk256.f32.f4.f4.f32
    {%d0...%d7}, %a, %b, {%c0...%c7};
```

- **m64nNk256**: M=64 warps, K=256 FP4 elements (256×4/8 = 128 bytes)
- N 可选: 8~256 (步长 8)
- 需要 TMA (Tensor Memory Accelerator) 加载
- **Scale 对齐**: K=256 = 16 个 sub-block, 需要 16 个 FP8 缩放因子

### §9.3 TCGEN05 (Threadblock-level GEMM)

```
tcgen05.mma.cta_group::1.m64nNk256.Kind.f4.f4.accum
```

- CTAGROUP=1: 单 CTA group
- 自动管理 shared memory → register 数据流
- 最高吞吐: 整个 threadblock 协作完成大规模 FP4 GEMM

### §9.4 PTX ISA 版本

- PTX ISA 12.x / 13.x 对 `.f4` 类型的 MMA 语法和寄存器分配进行了细化
- 需要 `--gpu-architecture=sm_100` 或更高编译目标
- Scale 寄存器布局: 按 sub-block 顺序排列在独立的 register group 中

## §10 gllm JIT Codegen 集成

### §10.1 VmInstr 映射

| 操作 | VmInstr | 说明 |
|------|---------|------|
| 加载 UE4M3 子块缩放 | `ScalarByteLoad` | 加载 1 字节 UE4M3 值 |
| 解码 E2M1 nibble × UE4M3 → FP32 | `Nvfp4SubBlockDequant` | 16 元素子块内解码 |
| 累加 | `Fma` | FP32 乘加 |
| 全局缩放 | 外层 `VecMulScalar` | F32 global tensor scale |

### §10.2 x86 实现 (AVX-512)

UE4M3 解码: GPR 整数运算 (extract exp/mant → 整数计算 power → XMM 转换)
E2M1 nibble 解码: VPBROADCASTD 加载 16 元素 LUT → VPSHUFB 查表
乘法: VMULPS (E2M1 × UE4M3 scale)

### §10.3 AArch64 实现 (NEON/SVE)

UE4M3 解码: UBFM/ADD/LSL/ORR 提取 exp/mant → 整数乘加
E2M1 nibble 解码: 复用 MXFP4 的 `emit_mxfp4_dequant_4nibbles`
零值检查: CBZ + label patch

### §10.4 GPU 实现 — 双路径 (DeviceProfile 驱动)

GPU PTX codegen 必须根据 `DeviceProfile.tensor_cores_gen` 选择路径:

#### 路径 A: SM100+ 原生 FP4 Tensor Core (硬件原生, 零软件解码)

当 `DeviceProfile` 检测到 SM100+ 时, 必须直接使用 PTX MMA .f4 指令:

- Weight Loading 阶段按 §6.3 的 Split-Plane 布局准备两块独立内存:
  - `uint32_t* weight_ptr` — 纯 FP4 打包权重
  - `__nv_fp8_e4m3* scale_ptr` — 子块缩放矩阵
- 16 元素子块 → 1 次 `mma.sync.m16n8k32.f32.f4.f4.f32`
- Tensor Core 内部在计算流水线上同步完成 E2M1 解码 × UE4M3 缩放 × 乘加, 零软件位移
- 全局 F32 tensor scale 在 Epilogue 阶段应用 (`VecMulScalar`)
- 需要 `cuda_fp4.h` 头文件 + `--gpu-architecture=sm_100` 编译目标 + PTX ISA 12.x/13.x

**禁止**: SM100+ 硬件上走软件解码路径 — 这是性能浪费, 违反原生量化铁律 (SPEC/23 §1.1)

#### 路径 B: SM<100 软件解码 (兼容路径)

- **PTX**: 位操作提取 exp/mant → `ex2.approx.ftz.f32` 计算 2^(exp-7) → 软件 E2M1 LUT + UE4M3 解码 → `mma.sync` FP16/BF16
- **HIP**: union 位 reinterpret + 标量浮点运算
- **Metal**: `as_type<float>()` 位转换

## §11 参考

- NVIDIA Blackwell Architecture Guide: FP4 Tensor Core 规格
- GGML type 40: `ggml.h` `GGML_TYPE_NVFP4 = 40`
- llama.cpp `ggml/src/ggml.c`: `block_nvfp4` 结构体 + 解量化实现
- OCP Microscaling Formats (MX) Specification: E2M1 编码定义
- OCP FP8 Specification (OCP-OFP8): E4M3/E5M2 编码定义
- CUDA Toolkit 13.0+: `cuda_fp4.h` intrinsics
- PTX ISA 12.x/13.x: `.f4` MMA 指令语法与寄存器分配
