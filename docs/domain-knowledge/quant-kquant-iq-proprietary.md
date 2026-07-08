# K-Quant + IQ + 专有量化 领域资料库

> 来源：llama.cpp ggml-common.h（block struct）+ ggml-quants.c（encode/decode）+ OCP Microscaling Formats (MX) v1.0 spec（MXFP4/NVFP4 E2M1/E8M0/UE4M3）+ AWQ/GPTQ 论文 + 项目 `gllm-kernels/src/quant_format.rs`（register_kquant/register_iquant/register_external）+ `macros/quant_primitive/{k_quant,iq_series,commercial}.rs` + `codebooks.rs`
> 建库触发：预防性建库（BCE-20260708-GGUF-QUANT-Q4_0-LAYOUT-MISMATCH 修复方向反转事件后，扩展覆盖所有非 Classic 量化格式，堵未来同类布局/码本/缩放误解）。当前 BUG-KNOWLEDGE 仅有 GPU 侧 NVFP6 GEMM 条目，CPU 侧 K-Quant/IQ/专有格式无历史 BCE，本库为预防资产
> 最后验证：2026-07-08

---

## 核心机制

### 格式总表（GGUF type id → 项目 QuantType）

| 类别 | 格式 | GGUF id | QuantType | block_size | block_bytes | bpw |
|------|------|---------|-----------|------------|-------------|-----|
| **K-Quant** | Q2_K | 10 | Q2K | 256 | 84 | 2.625 |
| | Q3_K | 11 | Q3K | 256 | 110 | 3.4375 |
| | Q4_K | 12 | Q4K | 256 | 144 | 4.5 |
| | Q5_K | 13 | Q5K | 256 | 176 | 5.5 |
| | Q6_K | 14 | Q6K | 256 | 210 | 6.5625 |
| | Q8_K | 15 | Q8K | 256 | 292 | 8.625（含 bsums，中间格式）|
| **IQ** | IQ1_S | 20 | IQ1S | 256 | 66 | 1.5625 |
| | IQ1_M | 21 | IQ1M | 256 | 78 | 1.75 |
| | IQ2_XXS | 22 | IQ2XXS | 256 | 66 | 2.0625 |
| | IQ2_XS | 23 | IQ2XS | 256 | 74 | 2.3125 |
| | IQ2_S | 24 | IQ2S | 256 | 82 | 2.5625 |
| | IQ3_XXS | 25 | IQ3XXS | 256 | 98 | 3.0625 |
| | IQ3_S | 26 | IQ3S | 256 | 110 | 3.3125 |
| | IQ4_NL | 27 | IQ4NL | 32 | 18 | 4.5（非线性码本）|
| | IQ4_XS | 28 | IQ4XS | 256 | 136 | 4.25 |
| **专有 INT4** | AWQ4 | 30 | AWQ4 | 128 | 72 | 4 |
| | GPTQ4 | 31 | GPTQ4 | 128 | 72 | 4 |
| | Squeeze | 32 | Squeeze | 256 | 130 | 3 |
| **Float4** | MXFP4 | 33 | Mxfp4{32} | 32 | 17 | 4（OCP E2M1）|
| | NVFP4 | 34 | Nvfp4 | 64 | 36 | 4.5（NVIDIA E2M1 + UE4M3）|
| **Ternary** | TQ1_0 | 35 | TQ1_0 | 256 | 54 | 1.6875 |
| | TQ2_0 | 36 | TQ2_0 | 256 | 66 | 2.0625 |

### K-Quant 分层 scale 架构（关键认知）

所有 K-Quant 用 **super-block = 256 元素**（`QK_K = 256`），内部再分 **mini-block**（Q4_K/Q5_K = 8 个 32 元素块；Q2_K/Q3_K = 16 个 16 元素块）。Scale 是**两级**：

- **super-block 级**：`d`（F16 scale）+ 可选 `dmin`（F16 min-scale，Q2_K/Q4_K/Q5_K 有）
- **mini-block 级**：6-bit 编码的 `(sc, m)` 对，**不是直接 scale 值**，而是 `d * sc` / `dmin * m` 的组合。Q4_K/Q5_K 用 `scales[12]`（`K_SCALE_SIZE=12`）数组存 8 对 `(sc, m)`，通过 `get_scale_min_k4(j, scales)` 解码：

```c
// llama.cpp get_scale_min_k4 (ggml-quants.c), 项目 k_quant.rs:14-23 一致实现
j < 4:  sc = scales[j] & 63,          m = scales[j+4] & 63
j >= 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
        m  = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)
```

Q4_K block struct（llama.cpp + 项目一致）：`dm{d,dmin}` (4B) + `scales[12]` (12B) + `qs[128]` (128B) = 144B。

### K-Quant Q4_K nibble 布局 = SPLIT（与 Q4_0 一致）

Q4_K/Q5_K/Q6_K 的 nibble packing **沿用 Classic Q4_0 的 SPLIT 布局**（参见 `gguf-classic-quant-layout.md`）。证据：llama.cpp `dequantize_row_q4_K`（项目 `k_quant.rs:38-51` 一致）：

```c
// 每 32 元素块 (get_scale_min_k4(is))：
for (l in 0..32) y[...] = d1 * (q[l] & 0xF) - m1;   // 低 nibble → 前 32 元素
for (l in 0..32) y[...] = d2 * (q[l] >> 4)  - m2;   // 高 nibble → 后 32 元素
```

即 byte 的低 nibble 对应前半段、高 nibble 对应后半段（mini-block 粒度的 SPLIT）。**项目 `k_quant.rs` Q4_K 实现已正确对齐 llama.cpp**（含 `get_scale_min_k4` + SPLIT）。

### IQ 非线性码本量化（核心：必须查表）

IQ 系列不用线性 `d * q`，而是**码本查表**（lookup grid）。qs 存的是 **grid 索引**（不是数值），通过 grid 表查出 8 元素向量 + 符号位还原。

- **IQ4_NL**（最简 IQ）：码本 `kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10, 1, 13, 25, 38, 53, 69, 89, 113}`（llama.cpp `ggml-common.h`，项目 `quant_format.rs:1018` 一致）。4-bit nibble 是**码本下标**，decode = `d * kvalues_iq4nl[nibble]`。block 布局同 Q4_0（d + qs[16] = 18B），但 nibble 语义不同
- **IQ2/IQ3 系列**：用 `iq2xxs_grid[256]` / `iq2xs_grid[512]` / `iq3xxs_grid[256]` / `iq3s_grid[512]` 等 u64/u32 grid，每条目编码 8 或 4 个 int8 向量；配合 `ksigns_iq2xs[128]` 符号表
- **IQ1_S**：`iq1s_grid[2048]`（u64，每 byte ∈ {0x00, 0x01, 0xFF} = 三值 {-1,0,+1}），加 `IQ1S_DELTA = 0.125` 偏移
- **IQ1_M**：复用 `iq1s_grid`，但 scale 用 `iq1m_scale_t` 联合体（16 位分散在 `scales[]` 数组高位）
- **IQ4_XS**：复用 `kvalues_iq4nl` 码本，但 super-block 256 元素带 per-32-block 6-bit scale

### 外部专有 INT4（AWQ4 / GPTQ4 / Squeeze）

三者都是 **per-group 线性量化**（非码本），group_size=128（AWQ/GPTQ）或 256（Squeeze）：

- **AWQ4 block (128 元素, 72B)**：`scales` (F16, 2B) + pad(2B) + `zeros` (F16, 2B) + pad(2B) + `qweight[32]` (u32×32=128B... 注：descriptor block_bytes=72 与 qweight 64B 一致，32 u32 word × 8 nibble = 256 nibble 但 block_size=128 → 每 word 8 nibble 覆盖 8 元素，32 word × 4 元素冗余? 实际项目 commercial.rs:14 `for w in 0..32 { for nib in 0..8 }` = 256 元素，与 block_size=128 描述符不符，待核验)
- **GPTQ4**：布局同 AWQ4，但 `storage_layout = ColInterleaved`（g_idx 列重排后）
- **Squeeze**：3-bit 打包（256 元素 in 96 字节），`d*(q3 - 4)` 范围 [-4d, +3d]

### Float4：MXFP4 vs NVFP4（易混淆，重点）

两者都用 **E2M1**（4-bit float：2 exp + 1 mantissa + 1 sign）做数据，但**缩放机制完全不同**：

| 维度 | MXFP4（OCP 标准）| NVFP4（NVIDIA）|
|------|------------------|----------------|
| block_size | **32** | **64** |
| block_bytes | 17（1 scale + 16 data）| 36（4 scale + 32 data）|
| scale dtype | **E8M0**（纯指数 `2^(b-127)`，1 字节共享）| **UE4M3**（unsigned FP8 E4M3，**per-16 sub-block**）|
| scale 粒度 | per-block（32 元素共享 1 个 scale）| per-sub-block（每 16 元素 1 个 scale，共 4 个）|
| 缩放层级 | **一级**：`scale × e2m1[qs]` | **二级**：`global_f32 × ue4m3[sub] × e2m1[qs]` |
| E2M1 码本 | `kvalues_mxfp4[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}`（高 bit 表符号）| 同 MXFP4（共享 E2M1 编码）|
| native ISA | 无标准化（软件解）| SM100+ Blackwell FP4 tensor core（`Tcgen05Fp4`）|

E2M1 decode：nibble 低 3 bit 查 `kvalues_mxfp4`（绝对值），高 bit 决定符号（项目 `codebooks.rs` + llama.cpp `dequantize_row_mxfp4/nvfp4` 一致）。

### Ternary（TQ1_0 / TQ2_0）

- **TQ1_0**（1.6875 bpw）：**5-trit-per-byte** 编码（3^5=243，ceiling division），不是 nibble packing！256 元素用 52 字节 qs。decode = `d * (trit - 1)`，trit ∈ {0,1,2} → 值 {-d, 0, +d}（BitNet b1.58 三值权重）
- **TQ2_0**（2.0625 bpw）：2-bit packed ternary，每 byte 存 4 个 2-bit trit，decode = `d * (q2 - 1)`。64 字节 qs 覆盖 256 元素

---

## AI 易误判点（核心价值 — 堵幻觉）

### 易误判点 1：K-Quant 不是单一 scale，是两级缩放

- ❌ 误判：Q4_K 像 Q4_0 一样用一个 `d` 缩放整块
- ✅ 正解：Q4_K 有 **super-block d/dmin (F16)** + **8 个 mini-block 各自的 (sc, m) 对**（6-bit 编码在 `scales[12]`）。decode 必须 `get_scale_min_k4(j, scales)` 解出每 mini-block 的 `(sc, m)`，再 `d_eff = d * sc`, `m_eff = dmin * m`, `value = d_eff * nibble - m_eff`
- 证据：llama.cpp `dequantize_row_q4_K` + 项目 `k_quant.rs:14-51`；`K_SCALE_SIZE = 12` 容纳 8 对 6-bit (sc,m)
- 影响：漏掉 mini-block scale 用单一 d 解 → 数值系统性偏差

### 易误判点 2：K-Quant Q4_K/Q5_K/Q6_K nibble 也是 SPLIT（继承 Classic）

- ❌ 误判：K-Quant 用不同 nibble 编排
- ✅ 正解：Q4_K/Q5_K/Q6_K 的 qs nibble **沿用 Q4_0 的 SPLIT 布局**（低 nibble = 前半段，高 nibble = 后半段，mini-block 粒度）
- 证据：llama.cpp `dequantize_row_q4_K` 的 `q[l] & 0xF`（前 32）+ `q[l] >> 4`（后 32）；项目 `k_quant.rs:38-51` 一致
- 关联：参见 `gguf-classic-quant-layout.md` 的 SPLIT 布局定义。**项目 K-Quant 实现是正确的**（与 llama.cpp 对齐），不同于 classic.rs 的 INTERLEAVED 偏离

### 易误判点 3：IQ4_NL nibble 是码本下标，不是线性值

- ❌ 误判：IQ4_NL 的 nibble 像 Q4_0 一样 `value = d * (nibble - 8)`
- ✅ 正解：IQ4_NL 的 nibble 是 **`kvalues_iq4nl[16]` 码本的下标**，`value = d * kvalues_iq4nl[nibble]`。码本非线性：`{-127,-104,-83,-65,-49,-35,-22,-10, 1, 13, 25, 38, 53, 69, 89, 113}`（非均匀间隔）
- 证据：llama.cpp `dequantize_row_iq4_nl` 用 `kvalues_iq4nl[qs[j] & 0xf]`；项目 `quant_format.rs:1018` + `data_kind: QuantDataKind::PackedInt4` 但配 `codebook: Some(&IQ4_NL_CODEBOOK)`
- 影响：把 IQ4_NL 当 Q4_0 解 → 完全错误（线性 vs 非线性码本）

### 易误判点 4：NVFP4 是 per-16 sub-block 两级缩放，MXFP4 是 per-block 单级

- ❌ 误判：NVFP4 和 MXFP4 都是「一个 scale 缩放整块 E2M1」
- ✅ 正解：
  - **MXFP4**（block=32）：1 个 **E8M0** scale（`2^(b-127)`）缩放全块 32 个 E2M1 值
  - **NVFP4**（block=64）：**4 个 UE4M3 sub-block scale**（每 16 元素一个）+ 全局 F32 scale，**两级**：`global × ue4m3[sub] × e2m1[qs]`
- 证据：llama.cpp `block_mxfp4 { uint8_t e; uint8_t qs[16] }` vs `block_nvfp4 { uint8_t d[4]; uint8_t qs[32] }`（d[4] = 4 个 sub-block scale）；项目 `quant_format.rs:858 E8M0` vs `877 SubBlockScalars{sub_block_size:16, F8E4M3}`
- 影响：NVFP4 漏掉 sub-block scale 粒度 → 75% 元素用错 scale；MXFP4 误加 sub-block → 多余操作

### 易误判点 5：E2M1 码本是 `kvalues_mxfp4`，符号在高位

- ❌ 误判：E2M1 nibble 直接当 4-bit 有符号整数
- ✅ 正解：E2M1 用码本 `kvalues_mxfp4[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}`。低 3 bit 查绝对值（0/1/2/3/4/6/8/12），**bit 3 决定符号**（前 8 正，后 8 负）
- 证据：llama.cpp `dequantize_row_mxfp4` 用 `kvalues_mxfp4[qs[j] & 0x0F]` + `kvalues_mxfp4[qs[j] >> 4]`；`ggml-common.h` `kvalues_mxfp4` 定义

### 易误判点 6：TQ1_0 是 5-trit-per-byte，不是 nibble/2-bit

- ❌ 误判：TQ1_0 的 qs 是 2-bit packed（像 TQ2_0）或 nibble packed
- ✅ 正解：TQ1_0 用 **5-trit-per-byte** 编码（每 byte 容纳 5 个 base-3 数字，3^5=243，用 ceiling division `q = (q*256 + 242) / 243` 映射）。52 字节 qs 覆盖 256 元素（分 32-byte + 16-byte 两段不同打包密度）。项目 `quant_format.rs:902` 的 `data_layout: PackedNibbles` **描述符与实际不符**（TQ1_0 不是 nibble packing）
- 证据：llama.cpp `quantize_row_tq1_0_ref` / `dequantize_row_tq1_0` 用 `q *= 3; q += xi` 五次迭代 + ceiling division；项目 descriptor 标 `Packed` 但 data_layout 误标 PackedNibbles
- 影响：把 TQ1_0 当 nibble 解 → 完全乱码

### 易误判点 7：项目 AWQ4 用静态 q-8，GPTQ4 用 per-block q-zero（不对称）

- ❌ 误判：AWQ4 和 GPTQ4 解码公式一样
- ✅ 正解（项目当前实现 `commercial.rs`）：
  - **AWQ4**：`value = d * (nibble - 8.0)`（静态零点 8，**不读 block.zeros**）— `commercial.rs:18`
  - **GPTQ4**：`value = d * (nibble - zero)`，`zero = block.zeros & 0xF`（动态零点）— `commercial.rs:47,54`
- 矛盾：AWQ4 descriptor（`quant_format.rs:805`）声明 `ZeroLayout::BlockScalar{offset:4}`（有 per-block zero），但 scalar impl 不读它 → **descriptor 与 impl 不一致**，疑似 AWQ4 impl 遗漏 zero 字段（标准 AWQ 是 `d*(q-zero)`）。待核验是否 BUG
- 证据：对比 commercial.rs AWQ4（line 7-22，无 zero）vs GPTQ4（line 43-58，有 zero）

### 易误判点 8：项目 iq_series.rs IQ1_S 是 "simplified" 实现（疑似数值错误）

- ❌ 误判：项目 IQ1_S scalar impl 与 llama.cpp 对齐
- ✅ 正解：`iq_series.rs:7-50` 的 IQ1_S decode 注释明确写 "simplified: use byte pairs as index"（line 18）和 "Remaining 16 values (simplified: reuse grid pattern)"（line 43）。实现用同一 `grid_val` 填充 group 内全部 32 值（含复用 pattern 填后 16），**与 llama.cpp 完整实现差距大**：
  - llama.cpp IQ1_S：用 `iq1s_grid[qs | (qh<<8 & 0x700)]`（11-bit 索引，qh 提供高 3 位）+ `IQ1S_DELTA` 偏移 + scale 从 `qh[ib] >> 12 & 7` 提取 + shift bit 决定 ±delta
  - 项目：用 11-bit 索引但 qh 处理简化，无 delta，后 16 元素直接复用前 8 的 grid pattern
- 风险：IQ1_S 推理路径若走此 impl，**输出数值错误**。需用真实 .gguf 文件数值验证（类似 Q4_0 classic.rs 的循环论证风险）。**暂未触发 BCE，但高风险**
- 建议：使用 IQ1_S 前必须验证，或参照 llama.cpp `dequantize_row_iq1_s` 重写

### 易误判点 9：Q8_K 是中间格式（带 bsums），不是终端存储

- ❌ 误判：Q8_K 像 Q8_0 一样是终端权重量化格式
- ✅ 正解：Q8_K 是**中间量化格式**，用于 dot product 加速（权重先量化到 Q8_K，配合 Q2_K-Q6_K 做量化 GEMM）。block 含 `bsums[16]`（每 16 元素一组的有符号和），用于 `dot_qK_q8_K` 加速
- 证据：llama.cpp `block_q8_K { float d; int8_t qs[256]; int16_t bsums[16] }`，`quantize_row_q8_K_ref` 累加 bsums；项目 `k_quant.rs:92-99` Q8_K decode 仅 `d * qs[i]`（未用 bsums，正确，bsums 仅 dot 用）
- 影响：把 Q8_K 当终端格式 → 误以为模型权重会存 Q8_K（实际只在量化 GEMM 中间步骤出现）

### 易误判点 10：IQ1S_GRID 字节是三值 {0x00,0x01,0xFF}，不是任意 int8

- ❌ 误判：iq1s_grid 的 u64 条目可以任意解释为 8 个 int8
- ✅ 正解：IQ1S_GRID 每 u64 条目的 8 个 byte **只能是 0x00 / 0x01 / 0xFF**（对应三值 0/+1/-1，这是 BitNet-style 1-bit 量化）。项目 `codebooks.rs:1323` 测试断言此约束
- 证据：llama.cpp `iq1s_grid` 构造（`kgrid_1bit_2048` + `(pg[i]-1)/2` 映射到 {-1,0,+1}）；项目 `codebooks.rs` IQ1S_GRID 不变量测试

---

## 解决问题时参考

### 诊断 K-Quant 数值问题

1. 查 super-block d/dmin 是否正确读取（F16，2B 各）
2. 查 `get_scale_min_k4(j, scales)` 6-bit (sc,m) 解码是否对齐 llama.cpp（项目 `k_quant.rs:14-23` 已对齐）
3. 查 nibble 是否 SPLIT（低=前半，高=后半，mini-block 粒度）— 参见 `gguf-classic-quant-layout.md`
4. 查 mini-block 边界（Q4_K: 8 个 32 元素块；Q2_K: 16 个 16 元素块）

### 诊断 IQ 数值问题

1. 确认 grid 表是否完整加载（项目 `codebooks.rs`：IQ1S_GRID[2048]、IQ2XXS_GRID[256] 等）
2. 确认 qh 高位是否参与索引（IQ1_S/IQ1_M 的 11-bit 索引需 qh 提供 bit 8-10）
3. 确认符号表（`ksigns_iq2xs[128]`）+ mask（`kmask_iq2xs[8]`）正确使用
4. **IQ1_S 警告**：项目 impl 是 simplified 版，数值验证前不要信任
5. IQ4_NL/XS 必须查 `kvalues_iq4nl[16]` 码本，不能当线性 nibble

### 诊断 Float4（NVFP4/MXFP4）数值问题

1. **先判 block_size**：32 → MXFP4，64 → NVFP4
2. MXFP4：1 个 E8M0 scale（`2^(b-127)`）+ E2M1 码本查表（`kvalues_mxfp4`，bit3=符号）
3. NVFP4：4 个 UE4M3 sub-block scale（每 16 元素）+ 全局 F32 + E2M1 码本。**两级缩放顺序**：`global × ue4m3[sub] × e2m1[nibble]`
4. GPU 路径：NVFP4 走 SM100+ Tcgen05Fp4 tensor core（`native_isa: Some(Tcgen05Fp4)`），CPU 路径软件解

### 诊断 AWQ/GPTQ 数值问题

1. 确认 AWQ4 是否该用 per-block zero（descriptor 声明有，impl 没用）— 待核验
2. GPTQ4 storage 是 `ColInterleaved`（g_idx 列重排），AWQ4 是 `RowMajor`
3. group_size 非标准时（如 64/-1）需特殊处理

### 权威源查询路径

| 需求 | 查哪里 |
|------|--------|
| block struct 定义 | llama.cpp `ggml/src/ggml-common.h` |
| encode/decode 参考实现 | llama.cpp `ggml/src/ggml-quants.c`（`quantize_row_*_ref` / `dequantize_row_*`）|
| grid/码本表 | llama.cpp `ggml/src/ggml-quants.c`（`iq*grid` / `kvalues_*` / `ksigns_*`）|
| OCP MX spec（E2M1/E8M0/UE4M3）| https://www.opencompute.org/projects/microscaling-formats-mx |
| 项目 descriptor | `gllm-kernels/src/quant_format.rs`（register_*）|
| 项目 decode impl | `gllm-kernels/src/macros/quant_primitive/{k_quant,iq_series,commercial}.rs` |
| 项目码本 | `gllm-kernels/src/codebooks.rs` |
| NVFP4 GPU codegen | `gllm-kernels` GPU PTX path + `native_isa: Tcgen05Fp4` |
| GGUF dtype 枚举 + GgmlDType↔QuantType 映射 + row padding | [`gguf-format-spec.md`](./gguf-format-spec.md) §GgmlDType（36 variants 完整表）+ 易误判点#4（dtype↔quant 部分无映射，MXFP4 仅 block_size=32）+ §Row padding（block boundary 对齐）|

---

## 已知问题 / 边界

### 项目 AWQ4 descriptor/impl 不一致（待核验）

`quant_format.rs:805` AWQ4 声明 `ZeroLayout::BlockScalar{offset:4}`（per-block zero at offset 4），但 `commercial.rs:7-22` AWQ4 scalar decode 不读 `block.zeros`，用静态 `q - 8.0`。标准 AWQ 是 `d*(q-zero)`。可能是：
- (a) AWQ4 impl 遗漏 zero 字段（BUG）→ 应改为读 zero
- (b) 项目 AWQ4 约定 zero 已预编入（zero 恒为 8）→ 需 loader 配合
需 team-lead 核验后裁决。**未触发 BCE，预防性记录**。

### 项目 iq_series.rs IQ1_S 是 simplified 实现（高风险）

`iq_series.rs:7-50` 注释明示 "simplified"。与 llama.cpp `dequantize_row_iq1_s` 差距：缺 delta 偏移、qh 高位索引简化、后 16 元素复用 grid pattern。若 IQ1_S 模型走此 impl 推理，输出错误。**未触发 BCE，但数值验证前不可信**。建议参照 llama.cpp 重写或加数值对齐测试（用真实 IQ1_S .gguf 对比 Python transformers）。

### TQ1_0 descriptor data_layout 误标

`quant_format.rs:902` TQ1_0 的 `data_layout: PackedNibbles{offset:2, low_first:true}` 与实际 5-trit-per-byte 打包不符（`storage_layout: Packed` 是对的，但 data_layout 字段误用 PackedNibbles）。JIT 若按 PackedNibbles 语义生成解码会错。需改为专门的 Ternary5Trit 布局描述。

### K-Quant Q4_K/Q5_K SIMD 路径未实现

`k_quant.rs:59-78` Q4_K 的 avx2/avx512/neon decode 全部 fallback 到 scalar（注释 "Correctness-first: use scalar logic until SIMD is updated"）。性能未优化但数值正确。

### GPU NVFP4/NVFP6 GEMM 已实现（参考 BUG-KNOWLEDGE）

GPU 侧 NVFP4/NVFP6 GEMM 走 SM100+ Blackwell tensor core，BCE-20260703-GPU-NVFP6-GEMM-IMPL 已治本实现（真实 VmInstr + PTX emit）。CPU 侧 NVFP4 软件解路径独立。本库关注 CPU JIT 侧 + 内存布局，GPU codegen 见 BUG-KNOWLEDGE 对应条目。

### 参考资料版本

- llama.cpp 源码：master 分支（2026-07-08 抓取，与建库1同批次），K-Quant/IQ/TQ/MXFP4/NVFP4 block struct + encode/decode 稳定
- OCP MX v1.0 spec：E2M1/E4M3/E8M0 编码定义权威源
- 项目代码：`gllm-kernels/src/quant_format.rs`（descriptor + IQ4_NL_CODEBOOK）、`macros/quant_primitive/{k_quant,iq_series,commercial}.rs`（decode/dot impl）、`codebooks.rs`（IQ1S_GRID 等码本表）
- 关联资料库：
  - [`gguf-classic-quant-layout.md`](./gguf-classic-quant-layout.md) — Classic 6 格式 + SPLIT 布局定义（K-Quant Q4_K/Q5_K/Q6_K nibble 继承同 SPLIT 布局，项目 classic.rs INTERLEAVED 偏离对照）
  - [`gguf-format-spec.md`](./gguf-format-spec.md) — GGUF 文件格式 + 36 GgmlDType 完整表（含项目自定义 AWQ4/GPTQ4/SQUEEZE/NVFP4）+ GgmlDType↔QuantType 映射（易误判点#4，MXFP4 仅 block_size=32）+ row padding（量化 block 对齐规则）+ tensor 命名规范
