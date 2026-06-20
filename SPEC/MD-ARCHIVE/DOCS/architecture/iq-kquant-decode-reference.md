# IQ 系列 + K-Quant 低比特解码技术参考

> **用途**: 本文档为 gllm JIT codegen 实现 IQ1/IQ2/IQ3/Q2K/Q3K 的解码提供底层位域定义和解码算法。
> 涵盖代数格点 (Algebraic Lattice) 查找、多级位域缝合、分层缩放等底层细节。
> 实现规范见 `SPEC/23-QUANT-CODEGEN-ALGO.md`。

## §1 核心概念: 代数格点 (Algebraic Lattice)

IQ 系列量化 (Importance-matrix Quantization) 的核心区别于传统整数量化:

- **传统量化** (Q4_0/Q4K): 线性均匀量化, `value = (integer - bias) × scale`
- **IQ 量化**: 非均匀格点量化, `value = grid[index] × d × scale`

格点 (Lattice/Grid) 是预计算的离散值集合, 硬编码在算子中, 不需要运行时分配内存。
不同 IQ 变体使用不同的格点结构:

| 变体 | 格点类型 | 每元素比特 | 格点来源 |
|------|---------|-----------|---------|
| IQ1_S / IQ1_M | A8/E8 经典格点变体 | ~1 bit | 固定代数集合 ({-1, 0, 1} 加权组合) |
| IQ2_XXS / IQ2_XS / IQ2_S | 8 维格点 | ~2 bit | 硬编码常量向量 (如 `iq2xxs_grid`) |
| IQ3_XXS / IQ3_S | 3 维格点 | ~3 bit | 符号位 + 幅值的离散格点点集 |
| IQ4_NL | 显式非线性查找表 | 4 bit | 16 元素 codebook (`IQ4_NL_CODEBOOK`) |

格点幅值典型分布: ±0.5, ±1.0, ±1.5 等由格点基底线性组合的值。

## §2 IQ1_S 解码 (1-bit, block_size=256, 50 bytes)

### §2.1 数据结构

```
BlockIQ1S (50 bytes):
  d:     f16          [offset 0, 2 bytes]  — 超级块全局缩放
  qs:    [u8; 32]     [offset 2, 32 bytes] — 256 bits, 每元素 1-bit 基础索引
  qh:    [u16; 8]     [offset 34, 16 bytes] — 高位额外信息 / 符号选取
  scales: [u8; 16]    [offset 50, 16 bytes] — 16 子块缩放 (4-bit packed)
```

### §2.2 解码步骤

```
对每个元素 n (0..255):
  1. group = n / 16                        // 16 个子块
  2. qs_bit = (qs[n/8] >> (n%8)) & 1       // 基础 1-bit 索引
  3. qh_bit = (qh[...] >> ...) & 1         // 高位扩展 (A8/E8 格点坐标)
  4. index = combine(qs_bit, qh_bit)        // 组合为格点索引
  5. grid_value = a8_e8_grid[index]         // 格点查找
  6. scale = decode_scales(scales, group)   // 子块缩放
  7. value = grid_value × d × scale         // 最终值
```

## §3 IQ2_S 解码 (2-bit, block_size=256, 82 bytes)

### §3.1 数据结构

```
BlockIQ2S (82 bytes):
  d:      f16          [offset 0, 2 bytes]  — 超级块缩放
  qs:     [u8; 64]     [offset 2, 64 bytes] — 低 2-bit (每元素 4 状态)
  qh:     [u8; 8]      [offset 66, 8 bytes] — 高位 (每元素 0.25 bit)
  scales: [u8; 8]      [offset 74, 8 bytes] — 4-bit packed 子块缩放
```

### §3.2 解码步骤

```
对每个元素 n (0..255):
  1. qs_low2 = (qs[n*2/8] >> (n*2%8)) & 3   // 低 2-bit 基础索引
  2. qh_bit = (qh[n/32] >> (n%32)) & 1      // 高位 0.25-bit 扩展
  3. index = (qh_bit << 2) | qs_low2        // 组合为格点查找键
  4. grid_value = iq2s_grid[index]           // 格点查找
  5. group = n / 32                          // 子块分组
  6. scale = decode_4bit_scale(scales, group)
  7. value = grid_value × d × scale
```

**反量化公式**: `Value = grid_value × d × scale`

## §4 IQ3_XXS 解码 (3-bit, block_size=256, 98 bytes)

### §4.1 数据结构

```
BlockIQ3XXS (98 bytes):
  d:  f16          [offset 0, 2 bytes]  — 超级块缩放
  qs: [u8; 96]     [offset 2, 96 bytes] — 768 bits = 256 × 3 bits
```

### §4.2 3-bit packed 位域解包

qs[96] 中每 3 个 bit 表示一个元素。解包方式:

```c
// 每次从 uint32_t 中提取 4 个元素 (4 × 3 = 12 bits)
uint32_t packed = *(uint32_t*)(qs + byte_offset);
int q0 =  packed        & 0x7;
int q1 = (packed >>  3) & 0x7;
int q2 = (packed >>  6) & 0x7;
int q3 = (packed >>  9) & 0x7;
```

SIMD 优化版本: 并行算术移位直接解包为字节数组, 然后映射到小型对称代数格点空间。

### §4.3 解码步骤

```
对每个元素 n (0..255):
  1. 3bit_index = extract_3bit(qs, n)        // 3-bit 格点索引
  2. grid_value = iq3xxs_grid[3bit_index]    // 格点查找
  3. value = grid_value × d
```

## §5 K-Quant 6-bit Packed Scales 解码 (Q4K/Q5K 共用)

### §5.1 概述

`scales[12]` (96 bits) 存储 8 个 (scale, min) 对, 每个对占 6+6=12 bits。
核心函数: `get_scale_min_k4`。

### §5.2 位域缝合算法

```c
// 输入: q[12] 字节数组 (scales[12])
// 输出: d[8] (scales) + m[8] (mins)

// 块 0-3: 直接取低 6 位
d[0] = q[0]  & 0x3F;   // bits [5:0]  of q[0]
m[0] = q[4]  & 0x3F;   // bits [5:0]  of q[4]
d[1] = q[1]  & 0x3F;
m[1] = q[5]  & 0x3F;
d[2] = q[2]  & 0x3F;
m[2] = q[6]  & 0x3F;
d[3] = q[3]  & 0x3F;
m[3] = q[7]  & 0x3F;

// 块 4-7: 跨字节缝合
// 低 4 位来自 q[j+4], 高 2 位来自 q[j-4] 的高 2 位
d[4] = (q[4] & 0x0F) | ((q[0] >> 6) << 4);   // q[4] 低4 + q[0] 高2
m[4] = (q[8] & 0x0F) | ((q[4] >> 6) << 4);   // q[8] 低4 + q[4] 高2
d[5] = (q[5] & 0x0F) | ((q[1] >> 6) << 4);
m[5] = (q[9] & 0x0F) | ((q[5] >> 6) << 4);
d[6] = (q[6] & 0x0F) | ((q[2] >> 6) << 4);
m[6] = (q[10] & 0x0F) | ((q[6] >> 6) << 4);
d[7] = (q[7] & 0x0F) | ((q[3] >> 6) << 4);
m[7] = (q[11] & 0x0F) | ((q[7] >> 6) << 4);
```

### §5.3 最终缩放

```
对 Q4_K: 每个 sub-block i (32 elements):
  effective_scale = d_block × d[i]
  effective_min   = dmin_block × m[i]
  value = (nibble × effective_scale) - effective_min
```

## §6 Q2_K 解码 (2-bit, block_size=256, 84 bytes)

### §6.1 数据结构

```
BlockQ2K (84 bytes):
  scales: [u8; 16]  [offset 0, 16 bytes]  — 16 对 4-bit (scale, min)
  qs:     [u8; 64]  [offset 16, 64 bytes] — 512 bits = 256 × 2 bits
  d:      f16       [offset 80, 2 bytes]  — 超级块 scale
  dmin:   f16       [offset 82, 2 bytes]  — 超级块 min scale
```

### §6.2 scales[16] 的 4-bit 对

每个 byte 存储一对 (scale_index, min_index):

```
scales[i]:
  低 4 位 (sc & 0x0F): 小块 scale 索引
  高 4 位 (sc >> 4):   小块 min 索引
```

分层缩放:
```
dl = d × (scales[i] & 0x0F)    // 子块 scale
ml = dmin × (scales[i] >> 4)   // 子块 min
```

### §6.3 qs[64] 的 2-bit 解包

64 bytes × 8 bits = 512 bits = 256 × 2-bit 元素。

解包使用跨步复用: 超级块分为两组 (0~127 和 128~255), 位移步长每次增加 2:

```c
int shift = 0;
for (int j = 0; j < 4; ++j) {
    uint8_t sc = scales[is++];
    float dl = d * (sc & 0xF);
    float ml = dmin * (sc >> 4);
    // 处理 16 个元素:
    for (int i = 0; i < 16; ++i) {
        int qval = (qs[qs_idx] >> shift) & 0x3;
        value = dl * qval - ml;
    }
    shift += 2;
}
```

### §6.4 最终反量化

```
value = dl × (2-bit unsigned) - ml
      = d × (sc & 0xF) × qs_val - dmin × (sc >> 4)
```

## §7 Q3_K 解码 (3-bit, block_size=256, 110 bytes)

### §7.1 数据结构

```
BlockQ3K (110 bytes):
  hmask:  [u8; 32]   [offset 0, 32 bytes]  — 1-bit/elem 掩码
  qs:     [u8; 64]   [offset 32, 64 bytes]  — 2-bit/elem 低 bit
  scales: [u8; 12]   [offset 96, 12 bytes]  — Q3KExtended 缩放因子
  d:      f16        [offset 108, 2 bytes]  — 超级块 scale
```

### §7.2 位域组合

```
3-bit value = (qs 的 2-bit) | (hmask 的 1-bit 作为符号/高位)
```

hmask[32] 每字节服务 32 个元素中的 8 个 (1 bit/element)。
qs[64] 每字节服务 4 个元素 (2 bits/element)。

### §7.3 Q3KExtended 缩放因子解码

scales[12] (96 bits) 解码为 16 个 6-bit 缩放因子:

**低 4 位来源**:
- 前 8 块: scales[0..7] 的低 4 位
- 后 8 块: scales[0..7] 的高 4 位

**高 2 位来源**:
- scales[8..11] 每字节提取 4 个块的高 2 位 (4 × 2 = 8 bits/byte)

**组合**:
```
scale_6bit = low_4bit | (high_2bit << 4)
```

### §7.4 最终反量化

```
value = (3-bit_value - 4) × d × scale_6bit
```

StaticBias = 4 (已在注册中设置 `ZeroLayout::StaticBias { value: 4 }`)。

## §8 B 类精确结论与代码修正指引

### §8.1 MXFP4 E8M0 Scale — 必须修正注册

**结论**: 当前注册 `ScaleDType::F8E4M3` **错误**。MXFP4 使用 OCP MX 标准的 E8M0 格式。

E8M0 编码: 8-bit 纯指数, 无尾数 (M0 = 0 mantissa bits)。
```
scale = 2^(byte - 127)
```

| 示例 byte | 计算 | scale |
|-----------|------|-------|
| 127 | 2^(127-127) | 1.0 |
| 130 | 2^(130-127) | 8.0 |
| 120 | 2^(120-127) | 0.0078125 |

**代码修正**: `quant_format.rs` MXFP4 注册中 `ScaleDType::F8E4M3` → 需要新增 `ScaleDType::E8M0` 并使用。

### §8.2 Q8_K bsums — JIT GEMM 核心优化

**结论**: 强烈建议在 JIT 中利用 `bsums[16]`。

在 `Y = X · W` 中, 若权重 W 有非对称偏置 (形如 `q × d + m`), 展开乘法产生:
```
Σ(X_i × m) = m × Σ(X_i)
```

Q8_K 在量化阶段预计算每个 16 元素子块的和, 存入 `bsums[16]`。

**JIT 优化**: 读取 Q8_K 的 bsums 可将偏置乘法从 O(N) 循环降为 O(1) 查表乘法:
```
// 无 bsums: 每次乘法都要遍历激活值求和
for i in 0..N { acc += x[i] * m; }

// 有 bsums: 直接查表
bias = m * bsums[sub_block_idx];  // 一条乘法指令
```

### §8.3 Q4_1/Q5_1 — 显式 Min 无 StaticBias

**结论**: _1 系列有显式 min (m), 无 StaticBias。nibble/uint5 直接作为纯正整数参与计算。

| 格式 | 公式 | ZeroLayout |
|------|------|-----------|
| Q4_0 | `value = (nibble - 8) × d` | `StaticBias { value: 8 }` |
| Q4_1 | `value = nibble × d + m` | `BlockScalarWithMin { d, m }` (m 通常为负数) |
| Q5_0 | `value = (uint5 - 16) × d` | `StaticBias { value: 16 }` |
| Q5_1 | `value = uint5 × d + m` | `BlockScalarWithMin { d, m }` (m 通常为负数) |

**关键**: nibble/uint5 不减任何静态偏移, m 自动充当零点角色。
当前注册中 Q4_1/Q5_1 的 `ZeroLayout` 已经是正确的 (`BlockScalarWithMin` 无 `StaticBias`)。

### §8.4 SqueezeLLM — 必须加 Codebook

**结论**: SqueezeLLM 论文是 3-bit 非均匀聚类量化, GGUF 工程实现用 4-bit nibble 对齐存储, 但 **必须使用 Lookup Table (LUT)**。

- 论文核心: 3-bit 聚类 → 8 个聚类中心值
- 工程存储: 3-bit 填充到 4-bit nibble (避免跨字节 3-bit 移位的访存惩罚), 高位补 0
- Block 结构: 128 bytes data (4-bit aligned, 256 elements) + 2 bytes f16 scale = 130 bytes
- 反量化: `value = LUT[packed_val] × scale`

**代码修正**: 当前注册 `codebook: None` + `data_kind: PackedInt4` 需要修正:
- `data_kind` 应改为 `SuperLowBit` (非均匀 codebook 查找, 不是线性 INT4)
- `data_layout` 应改为 `CodebookIndex { offset: 2, index_bits: 4 }`
- `codebook` 应填入 8 个聚类中心的 LUT (3-bit 有效索引, 高位为 0)

## §9 参考

- llama.cpp `ggml/src/ggml.c`: IQ 系列 grid 定义 + Q2K/Q3K 解码实现
- llama.cpp `ggml/src/ggml-quants.c`: `get_scale_min_k4` 位域缝合
- A8/E8 Lattice: Conway & Sloane "Sphere Packings, Lattices and Groups"
- SqueezeLLM 论文: "SqueezeLLM: Dense-and-Sparse Quantization" (Kim et al., 2023)
- SPEC/23-QUANT-CODEGEN-ALGO.md: 量化 GEMM 微核三阶段 (Prologue + Compute + Epilogue)
