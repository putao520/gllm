# GGUF Classic GGML 量化布局 领域资料库

> 来源：llama.cpp ggml-common.h（block struct）+ ggml-quants.c（`quantize_row_*_ref` 编码器 / `dequantize_row_*` 解码器，GGUF 格式的权威参考实现）+ 项目 `gllm-kernels/src/macros/quant_primitive/classic.rs` + `gllm-kernels/src/quant_format.rs`
> 建库触发：`BCE-20260708-GGUF-QUANT-Q4_0-LAYOUT-MISMATCH`（Q4_0 prefill 输出乱码，JIT Assisted GEMM 路径 vs GEMV classic.rs 路径 nibble 编排分歧）
> 最后验证：2026-07-08

---

## ⚠️ 建库首要发现（必须先读）

**源码核验后发现 `BCE-20260708-GGUF-QUANT-Q4_0-LAYOUT-MISMATCH` 条目对 nibble 布局方向的结论是反的。** 作者itative llama.cpp 源码证明：

| 路径 | 真实布局（llama.cpp 权威） |
|------|---------------------------|
| 标准 GGUF / llama.cpp 参考实现（`dequantize_row_q4_0` / `quantize_row_q4_0_ref`） | **SPLIT**：`qs[j]` 低 nibble → element `j`，高 nibble → element `j+16` |
| gllm `classic.rs` 标量解码（Q4_0/Q4_1/Q5_0/Q5_1） | **INTERLEAVED**：`qs[i]` 低 nibble → element `2i`，高 nibble → element `2i+1` |
| gllm JIT Assisted GEMM 注释（quant_gemm.inc.rs:347） | **SPLIT**（与 llama.cpp 一致） |

**结论反转**：gllm `classic.rs`（interleaved）偏离 GGUF 标准；JIT Assisted（split）符合标准。BCE 条目称 "JIT 误假设 split / classic.rs interleaved 是正解" 的方向需要由 team-lead 重新核验后再动代码——本资料库只记录事实，不替 team-lead 做改代码决策。

---

## 核心机制（6 种 Classic 格式，block_size 全部 = 32）

### block 结构（llama.cpp `ggml-common.h`，与 `quant_format.rs` 一致）

| 格式 | block_bytes | 结构（C struct） | scale/zero 类型 |
|------|-------------|------------------|------------------|
| **Q4_0** | **18** | `ggml_half d; uint8_t qs[16]` | StaticBias（固定零点 8，d=max/-8 有符号）|
| **Q4_1** | **20** | `ggml_half d; ggml_half m; uint8_t qs[16]` | BlockMin（per-block min，d=(max-min)/15 无符号）|
| **Q5_0** | **22** | `ggml_half d; uint8_t qh[4]; uint8_t qs[16]` | StaticBias（固定零点 16，d=max/-16 有符号 5-bit）|
| **Q5_1** | **24** | `ggml_half d; ggml_half m; uint8_t qh[4]; uint8_t qs[16]` | BlockMin（无符号 5-bit + per-block min）|
| **Q8_0** | **34** | `ggml_half d; int8_t qs[32]` | BlockScalar（signed INT8，d=amax/127）|
| **Q8_1** | **36** | `ggml_half d; ggml_half s; int8_t qs[32]` | BlockScalar + sum（s = d * sum(qs)，INT8）|

- `ggml_half` = IEEE 754 half（2 字节），即项目 `F16`
- Q5_0/Q5_1 的 `qh[4]` 是 32-bit little-endian，每位对应一个 element 的第 5 bit
- Q8_1 第二个 F16 llama.cpp 命名为 `s = d*sum(qs[i])`，**不是 min/offset**；项目 `quant_format.rs` 的 `BlockMin` 标签略有语义偏差，但 decode 时 Q8_1 复用 Q8_0 路径（`s` 字段仅在点积优化用），功能等价

### dequantize 公式（llama.cpp `ggml-quants.c` 权威）

- **Q4_0**: `y[j]   = ((qs[j] & 0x0F) - 8) * d`，`y[j+16] = (qs[j] >> 4) - 8) * d`（j=0..15）
- **Q4_1**: `y[j]   = (qs[j] & 0x0F) * d + m`，`y[j+16] = (qs[j] >> 4) * d + m`
- **Q5_0**: 5-bit 值 `q = (qs[j] 低 nibble) | (((qh >> j) & 1) << 4)`，`y[j]   = (q - 16) * d`；后半段 `y[j+16]` 用 `qs[j]` 高 nibble + `qh` 的第 `j+16` 位
- **Q5_1**: 同 Q5_0 但 `y = q*d + m`（无符号 + 偏移）
- **Q8_0**: `y[i] = qs[i] * d`（i=0..31，signed i8，无 nibble packing）
- **Q8_1**: decode 时 `y[i] = qs[i] * d`（复用 Q8_0，`s` 字段在 decode 不用）

### nibble 布局：SPLIT（llama.cpp 标准，GGUF 文件实际格式）

Q4_0/Q4_1/Q5_0/Q5_1 共用同一种 nibble packing（`quantize_row_q4_0_ref` 编码 + `dequantize_row_q4_0` 解码一致）：

```
16 字节 qs[0..15] 编码 32 个 element：
  byte qs[j] (j=0..15):
    低 nibble (qs[j] & 0x0F)  → element[j]       (前半 0..15)
    高 nibble (qs[j] >> 4)    → element[j+16]    (后半 16..31)

例: qs[0] = (elem[0], elem[16])，qs[1] = (elem[1], elem[17])，...，qs[15] = (elem[15], elem[31])
```

Q5_0/Q5_1 的 5th bit 同理 SPLIT：`qh` 的 bit `j` 对应 element `j`，bit `j+16` 对应 element `j+16`。

Q8_0/Q8_1 无 nibble packing（每 element 占 1 字节 signed i8，`qs[i] → element[i]` 顺序排列），**不存在 split/interleaved 歧义**。

---

## AI 易误判点（核心价值 — 堵幻觉）

### 易误判点 1（最关键）：Q4_0 nibble 布局方向

- ❌ **误判**：Q4_0 的 16 字节 qs[] 是「interleaved」排列——byte i 的低 nibble = element[2i]，高 nibble = element[2i+1]（相邻 element 对共享一个 byte）
- ✅ **正解**：Q4_0 的 16 字节 qs[] 是 **SPLIT** 排列——byte j 的低 nibble = element[j]，高 nibble = element[j+16]（前半段 16 个 nibble 在所有 byte 低 4 位，后半段 16 个 nibble 在所有 byte 高 4 位）
- **证据**：llama.cpp `ggml-quants.c` `dequantize_row_q4_0` 明确写 `y[i*qk + j + 0] = x0*d`（low→elem j）和 `y[i*qk + j + qk/2] = x1*d`（high→elem j+16）；编码器 `quantize_row_q4_0_ref` 同样 `x0 = x[0+j]`，`x1 = x[qk/2+j]`
- **影响**：用错布局解码 GGUF 文件 → 权重乱码 → 推理输出全错（prefill argmax 飘到随机 token）

### 易误判点 2：Q4_0 是有符号 nibble，Q4_1 是无符号 nibble

- ❌ 误判：所有 4-bit 经典量化都是「nibble 直接乘 scale」
- ✅ 正解：
  - **Q4_0 / Q5_0**（type-0）= **有符号**：nibble 值减固定零点（Q4_0 减 8，Q5_0 减 16），`d = max / -8`（d 含符号）
  - **Q4_1 / Q5_1**（type-1）= **无符号 + per-block min**：nibble 值直接乘 d 再加 per-block `m`（min），`d = (max-min)/15`
- 证据：`quantize_row_q4_0_ref` 算 `d = max / -8`（符号在 d 里），nibble = `(int)(x*id + 8.5)`；`quantize_row_q4_1_ref` 算 `d = (max-min)/15`，nibble = `(int)((x-min)*id + 0.5)`

### 易误判点 3：Q5_0/Q5_1 的 qh 位序

- ❌ 误判：`qh[4]` 的 bit `i` 直接对应 element `i`，或者高低字节有特殊编排
- ✅ 正解：`qh` 是 32-bit little-endian，bit `j`（j=0..15）→ element `j` 的 5th bit，bit `j+16`（j=0..15）→ element `j+16` 的 5th bit（与 nibble SPLIT 布局一致，前半 bit 对应前半 element，后半 bit 对应后半 element）
- 证据：`quantize_row_q5_0_ref` 写 `qh |= ((xi0 & 0x10u) >> 4) << (j + 0)`（前半 element），`qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2)`（后半 element）；`dequantize_row_q5_0` 用 `xh_0 = ((qh >> (j+0)) << 4) & 0x10`，`xh_1 = ((qh >> (j+12)) ) & 0x10`（注意 j+12 是因为先左移 4 再右移，等价于 j+16）

### 易误判点 4：Q8_1 的第二 F16 是 `s`（d*sum(qs)）不是 `m`（min）

- ❌ 误判：Q8_1 像 Q4_1/Q5_1 一样用 min 偏移
- ✅ 正解：llama.cpp `block_q8_1` 第二 F16 是 `s = d * sum(qs[i])`（sum-of-quants × d），用于点积优化（`dot = d * sum(qs[i]*other[i])`，当 other 全 1 时退化），**decode 时不起偏移作用**
- 证据：`quantize_row_q8_1_ref` 累加 `sum += qs[i]` 最后 `s = sum*d`；gllm `classic.rs` Q8_1 decode 复用 Q8_0 路径（不读 s）
- 项目 `quant_format.rs` 把 Q8_1 第二 F16 标为 `BlockMin { offset_bytes: 2 }` 是**命名偏差**，功能上 Q8_1 decode 复用 Q8_0 不受影响，但若新路径依赖该标签语义需注意

### 易误判点 5：d 字段是 F16 不是 F32

- ❌ 误判：scale `d` 是 F32（4 字节）
- ✅ 正解：所有 6 种 Classic 格式的 scale `d`（以及 Q4_1/Q5_1 的 min `m`、Q8_1 的 `s`）都是 **IEEE 754 half precision（F16，2 字节）**
- 证据：`ggml-common.h` 全部 `block_q*_struct` 用 `ggml_half d`（= uint16_t，2 字节）；项目 `classic.rs` 用 `block.d.to_f32()` 转换
- 影响：偏移量计算必须按 F16 布局读（block_bytes 紧凑无 padding：Q4_0=2+16=18，Q4_1=2+2+16=20，Q5_0=2+4+16=22，Q5_1=2+2+4+16=24，Q8_0=2+32=34，Q8_1=2+2+32=36）

### 易误判点 6：「ggml/quants.c 注释 nibbles / quants」不等于 interleaved

- ❌ 误判：看到 `uint8_t qs[QK4_0/2]` 注释 "nibbles / quants" 就以为是相邻 element 交错打包
- ✅ 正解：「nibbles」指每 byte 存 2 个 4-bit 值，但**哪两个 element 共享一个 byte 由编码器决定**——llama.cpp 编码器选 SPLIT（前半/后半）
- 证据：必须读 `dequantize_row_*` 函数体的 `y[... + j + 0]` 和 `y[... + j + qk/2]` 索引才能确认编排方向，struct 定义 + 注释不足以下结论

---

## 解决问题时参考

### JIT dequantize / dot 实现（GEMV/GEMM Assisted 路径）

- **Q4_0/Q4_1/Q5_0/Q5_1 nibble 解码必须按 SPLIT**：byte j 的低 nibble → element[j]，高 nibble → element[j+16]。若 SIMD 解码后是「16 个低 nibble 连续 + 16 个高 nibble 连续」，那已经天然是 element[0..15] + element[16..31] 的正确顺序（SPLIT 的 SIMD 友好形态），**不需要额外 interleave/unpack 重排**
- **Q8_0/Q8_1 无 nibble packing**，`qs[i] → element[i]` 顺序，直接 INT8 SIMD（AVX2 `_mm256_cvtepi8_epi32` / VNNI `_mm256_dpbusd_epi32`）
- **Q4_0/Q5_0 有符号**：解码后减固定零点（8 / 16）；**Q4_1/Q5_1 无符号+min**：解码后乘 d 加 m
- **scale 读取**：block 起点的 F16（2 字节），`block.d.to_f32()`

### 验证布局方向的最快方法

写一个已知 GGUF 文件（如 SmolLM2-135M-Q4_0）的第一个权重 block，对照 llama.cpp `quantize_row_q4_0_ref` 反推：读 raw bytes → 用 SPLIT 解码 → 看是否与参考实现数值一致。**不要只信项目内 `test_dequant_q4_0_known_values` 这类测试**——它的 known values 可能由被测的（可能错的）解码器自生成（循环论证）。

### block 边界对齐

- GGUF 权重行长度必须是 `block_size=32` 的整数倍（否则 padding 或拒绝加载）
- 行 stride（字节）= `n_elem / 32 * block_bytes`（Q4_0: n_elem/32*18，Q8_0: n_elem/32*34）

### CPU SIMD 友好形态（SPLIT 的优势）

- SPLIT 布局下，16 个低 nibble（element 0..15）和 16 个高 nibble（element 16..31）分别在所有 byte 的低/高 4 位 → SIMD 可用 `_mm_and_si128(mask_0F)` + `_mm_srli_epi16(_, 4)` 一次性分离前半/后半，无需 interleave
- gllm `classic.rs` Q4_0 AVX2 实现用 `_mm256_unpacklo_ps(rl0, rh0)` interleave——**这暗示它假设的是 interleaved 布局**（与 llama.cpp SPLIT 冲突，需 team-lead 核验）

---

## 已知问题 / 边界

### gllm `classic.rs` 与 llama.cpp 标准的分歧（待 team-lead 裁决）

- `classic.rs` Q4_0/Q4_1/Q5_0/Q5_1 标量解码用 **interleaved**（`out[i*2]=lo, out[i*2+1]=hi`），与 llama.cpp **SPLIT** 标准不一致
- 两种可能（需 team-lead 验证哪种成立）：
  1. **gllm classic.rs 是 BUG**（偏离 GGUF 标准，读真实 .gguf 文件会乱码）→ 根治方向：把 classic.rs 改成 SPLIT（与 JIT Assisted 对齐），**不要**把 JIT 改成 interleaved
  2. **gllm loader 在加载时把 GGUF SPLIT 重排成内部 interleaved**（loader 转码）→ 则 classic.rs 对内部 blob 正确，JIT Assisted 读同一 blob 用 split 错；但 loader 转码无证据（需查 loader 代码确认），且这种"自创内部格式"违背项目 ARCH-BLOB-YIELDS-WEIGHT（权重 blob 须顺从权重文件原始布局）
- **高度怀疑是情况 1**（classic.rs BUG），因为：loader 重排 nibble 是罕见且违宪设计；JIT 注释 quant_gemm.inc.rs:347 描述的是 SPLIT（与 llama.cpp 一致，像是作者查过标准）；prefill 乱码症状与 "GEMV 用 classic.rs interleaved 解码 SPLIT 的 GGUF blob" 吻合
- **本资料库不替 team-lead 做决策**，只提供事实：GGUF 标准 = SPLIT（llama.cpp 权威），gllm classic.rs = interleaved（偏离）

### Q8_1 标签语义偏差（非阻断）

- `quant_format.rs` Q8_1 用 `ZeroLayout::BlockMin` 命名第二 F16，但 llama.cpp 语义是 `s = d*sum(qs)`（不是 min）
- 功能上 Q8_1 decode 复用 Q8_0 不读该字段，不引发数值错误；但若新路径按 BlockMin 语义使用会错

### Q4_0 的 d 符号约定

- `d = max / -8`（max 是绝对值最大的带符号值），所以 d 的符号与 max 相反；decode `y = (nibble - 8) * d` 中 `(nibble-8)` 范围 [-8, 7]，乘 d 后还原带符号权重
- 不要把 d 当无符号处理

### 参考资料版本

- llama.cpp 源码：master 分支（2026-07-08 抓取），Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 block struct 与 decode 逻辑多年稳定，向后兼容
- 项目代码：`gllm-kernels/src/macros/quant_primitive/classic.rs`（Q4_0 scalar line ~13-27, avx2 ~48-88; Q8_0 line ~816-891）、`gllm-kernels/src/quant_format.rs`（register_classic descriptors）、`gllm/BUG-KNOWLEDGE.md`（BCE-20260708 条目）
