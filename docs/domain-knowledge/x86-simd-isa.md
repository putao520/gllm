# x86 SIMD ISA 领域资料库（AVX2 / AVX-512 / VNNI / AMX / AVX10.2 / APX）

> 来源：Intel SDM Vol 2A + Intel Intrinsics Guide (v3.6.9) + WikiChip AVX-512 + Intel AMX Programming Guide + felixcloutier.com/x86
> 建库触发：16+ 同类 BCE 反复涉及 x86 SIMD 陷阱（AVX-512 半 lanes / BF16 VNNI guard / AMX tile / REGALLOC OOB / VCVTNEPS2BF16 lane-loss），从未统一文档化
> 最后验证：2026-07-08

---

## 核心机制（与出错相关的，源码 + 官方双确认）

### 1. 寄存器层级与别名 — 半 lanes 陷阱的物理根源

AVX-512 定义 **32 个 512-bit 向量寄存器 ZMM0…ZMM31**。AVX 的 16 个 YMM0…YMM15 是 ZMM0…ZMM15 的**低半别名**；SSE 的 16 个 XMM0…XMM15 是**最低 1/4 别名**。

**决定性规则**（WikiChip AVX-512 §Registers + §Integer/Floating-point instructions Common aspects，原文）：
> "If the destination is a vector register and the vector size is less than 512 bits, AVX and AVX-512 instructions **zero the unused higher bits** to avoid a dependency on earlier instructions writing those bits."

含义：**任何写 YMM/XMM 目标寄存器的指令，会自动把对应 ZMM 的更高位清零**。反之，用 YMM 寄存器名读一个 ZMM 时，只能读到低 256 位（lanes 0-7），高 256 位（lanes 8-15）读不到。

**补充**（WikiChip §Registers）：
> "many horizontal operations are confined to 128-bit lanes."

含义：水平归约（如 `vextractf128`）只能切到 128-bit 边界（lanes 4-7），**切不到 ZMM 的高 lanes**。

**这两条规则合起来 = AVX-512 半 lanes 陷阱**：在 W512 模式下用 YMM-only 算子（`scratch_ymm` + `ymmword_ptr(32B)` + `vextractf128`）只会处理 lanes 0-7，lanes 8-15 要么从未被读（归约类丢数据），要么被清零（写类产 NaN/垃圾）。

### 2. VEXTRACTF64X4 / VEXTRACTF32X4 — 取 ZMM 高半的唯一正确指令

WikiChip §Floating point instructions（原文）：
> "These instructions extract four (F32X4) or eight (F32X8) single precision values, or two (F64X2) or four (F64X4) double precision values … from a lane of that width (e.g. 128 bits for F32X4) of the source operand **selected by a constant index** and store the data in memory, or in the lowest lane of a vector register. **Higher lanes of the destination register are zeroed.**"

| 指令 | 扩展 | 宽度 | imm=0 | imm=1 |
|------|------|------|-------|-------|
| `VEXTRACTF64X4 ymm, zmm, imm` | AVX-512F | 512→256 | zmm 低 256 位 (f64 lanes 0-3) | **zmm 高 256 位 (f64 lanes 4-7)** |
| `VEXTRACTF32X4 xmm, zmm, imm` | AVX-512F | 512→128 | zmm lanes 0-3 | zmm lanes 4-7（imm=2/3 取 lanes 8-11/12-15） |
| `VEXTRACTI128 xmm, ymm, imm` | AVX2 | 256→128 | ymm 低 128 位 (lanes 0-3) | ymm 高 128 位 (lanes 4-7) |

**根治点**：要取 ZMM 高 8 个 f32 lanes（lanes 8-15），**唯一**正确做法是 `vextractf64x4(ymm_hi, zmm, 1)`（把高 256 位提到 ymm_hi），然后 `vmaxps(ymm_lo, ymm_lo, ymm_hi)` 把 16 lanes 归约成 8 lanes，再走 8→4→2→1 的 xmm reduce 链（`vextractf128` + `vmovhlps` + `vshufps`）。**不能用 `vextractf128`** — 它只能切 YMM 的 128-bit 边界，根本够不到 ZMM 高 lanes。

源码正确实现：`gllm-kernels/src/compiler/codegen/vm/x86_lower/lower_instr.inc.rs:977` `zmm_hreduce_to_xmm`（BCE-20260703-AVX512-HALF-LANES fixTemplate）。

### 3. VDPBF16PS（BF16 点积）+ 独立 CPUID 检测

WikiChip §Floating point instructions（原文）：
> "`VDPBF16PS`: Dot product of BFloat16 values, accumulated in single precision elements. The instruction multiplies the corresponding BF16 values of the source operands, converted to single precision, then adds the products from the **even lanes, odd lanes**, and the single precision values in the destination operand and stores the sums in the destination."

VNNI 语义：`dst[i] = dst[i] + src1[2i]*src2[2i] + src1[2i+1]*src2[2i+1]`（F32 累加）。

**扩展独立性**（WikiChip §Detection 权威表）：

| 特性 | CPUID 输入 | 输出位 | gllm-kernels flag |
|------|-----------|--------|-------------------|
| AVX512F | EAX=07H,ECX=0 | EBX[16] | `use_avx512` |
| AVX512CD | | EBX[28] | — |
| AVX512BW | | EBX[30] | — |
| AVX512DQ | | EBX[17] | — |
| AVX512VL | | EBX[31] | — |
| AVX512_VNNI | | ECX[11] | `has_vnni` |
| AVX512_BF16 | EAX=07H,**ECX=1** | EAX[5] | `has_bf16` |
| AVX512_FP16 | EAX=07H,ECX=0 | EDX[23] | `has_avx512fp16` |

**关键**：`use_avx512`（AVX512F）**不蕴含** `has_bf16` / `has_vnni` / `has_avx512fp16`。BF16/VNNI/FP16 是独立扩展特性，必须各自 CPUID 检测。

**实现支持矩阵**（WikiChip §Implementation，节选与 BF16/VNNI 相关）：

| 微架构 | VNNI | BF16 | AMX | 说明 |
|--------|------|------|-----|------|
| Skylake-X / Cascade Lake | ✘ | ✘ | ✘ | AVX-512 但重降频，无 AI 扩展 |
| Ice Lake (server) | ✔ | **✘** | ✘ | **有 VNNI 无 BF16** |
| Tiger Lake | ✔ | **✘** | ✘ | **有 VNNI 无 BF16** |
| Alder Lake | ✔ | ✔ | ✘ | AVX-512 实际被 Intel 禁用 |
| Sapphire Rapids | ✔ | ✔ | ✔ | 首个全支持（BF16+VNNI+AMX+FP16）|
| Granite Rapids | ✔ | ✔ | ✔ | + AMX-FP16 |
| AMD Zen 4 | ✔ | ✔ | ✘ | AMD 无 AMX |
| AMD Zen 5 | ✔ | ✔ | ✘ | 同上 |

**Ice Lake/Tiger Lake 有 AVX-512 VNNI 但无 BF16** — 这就是只查 `use_avx512` 就 emit `VDPBF16PS` 会 SIGILL 的根因（BCE-20260704-X86-BF16-VNNI-GUARD）。

### 4. VCVTNEPS2BF16 / VCVTNE2PS2BF16 — F32→BF16 转换的 lane-loss

WikiChip §Floating point instructions（原文）：
> "`VCVTNEPS2BF16` instruction uses one source operand, **writes only to the lower half of the destination vector and zeros the remaining elements**. The `VCVTNE2PS2BF16` instruction stores converted elements from the first source operand in the **upper half**, the second source operand in the **lower half** of the destination vector."

**关键**：`VCVTNEPS2BF16` 把 N 个 F32 转 N/2 个 BF16，**只写 dst 低半**（高半清零）。VecNarrow 一个 ZMM(16 F32) → YMM(16 BF16) 必须调用两次或在软件路径里用 `vextracti128` 显式处理高半，否则丢高 4-8 lanes（BCE-20260708-VECNARROW-LANE-LOSS）。

### 5. AVX-512 FP16（VADDPH / VFMADD213PH 等）

WikiChip §Floating point instructions：`V(ADD/DIV/MAX/MIN/MUL/SUB)PH` / `VF(MADD/MSUB)(132/213/231)PH` 由 AVX512_FP16 引入，CPUID EDX[bit 23]。仅 Sapphire Rapids / Granite Rapids / AMD Zen4+（部分）支持。FP16 是**原生计算**（不只是转换），区别于 F16C（只做 F16↔F32 转换，Ivy Bridge+ 基线）。

源码 `microarch.rs:139` `has_avx512fp16` 仅 `SapphireRapids | GraniteRapids`。

### 5b. FP8（AMX-FP8 / AVX10.2 FP8 转换，当前 ScalarLUT 软件路径）

FP8（E4M3 / E5M2）在 x86 上**不是单一 AVX-512 离散扩展**，分两条独立路径，且当前 gllm 支持的硬件（Sapphire Rapids / Granite Rapids / AMD Zen4-Zen5）**均无 FP8 硬件 dot product**：

| 路径 | 指令 | 硬件 | gllm 状态 |
|------|------|------|-----------|
| **AMX-FP8**（tile 矩阵乘） | `TDPFP8PS`（FP8×FP8→F32 tile 累加）| Diamond Rapids（未上市） | `microarch.rs:194` `has_amx_fp8` 预留 false |
| **AVX10.2 FP8 转换** | `VCVTFP8*` / `VCVT2FP8*`（F32↔FP8 转换，无原生 dot）| AVX10.2 conformance level | 走 ScalarLUT 软件解码（无硬件） |

**关键**：不存在 "AVX512_FP8" 独立 CPUID 扩展位。FP8 硬件能力要么在 AMX-FP8（tile 级 dot，Diamond Rapids+），要么在 AVX10.2（仅转换）。这与 BF16（AVX512_BF16，CPUID leaf7 subleaf1 EAX[5]）/ VNNI（AVX512_VNNI，leaf7 ECX[11]）/ FP16（AVX512_FP16，leaf7 EDX[23]）的"离散 AVX-512 扩展"模型**完全不同**。

**gllm-kernels FP8 实际路径**（dtype-propagation.md）：`FP8 → DequantCompute(ScalarLUT)` — FP8 权重查表解码为 F32，再走标准 F32 GEMM/FMA。microarch `has_amx_fp8=false`（所有当前 MicroArch 变体），codegen 无 FP8 硬件 dot 分支。

源码 `microarch.rs:178-196` `has_amx_fp16` / `has_amx_complex` / `has_amx_transpose` / `has_amx_fp8` 全预留 false（Diamond Rapids MicroArch 变体 TBD）。

### 6. AMX（Advanced Matrix Extensions）— TDPBF16PS

AMX 是 2D tile 寄存器 + TMUL（Tile Matrix Multiply Unit），独立于 AVX-512 的执行资源。

**启用前置条件**（Intel AMX Programming Guide + Optimization Reference Manual）：
1. **XCR0[18:17] = 11b** — bit 17 = TILECFG state，bit 18 = TILEDATA state。两者都要置位（AMX state 否则 `#UD`）
2. **CR4.AMXFLAG (CR4[18])** — 启用 AMX flag 管理
3. **LDTILECFG** — 加载 64-byte tile 配置结构（palette_id + 各 tile 的 rows/cols/stride/format）
4. **palette** — 当前 palette 0/1（Sapphire Rapids）；palette 2 在更新文档。每 palette 定义 tile 尺寸组合
5. **TILELOAD** / **TILESTORE** — 把数据装入 TMM0-7（8 个 tile 寄存器，每 tile 1KB = 16×16 BF16）
6. **TDPBF16PS tmm_dst, tmm_a, tmm_b** — 16×16×16 BF16 矩阵乘累加到 F32 tile

**仅 Sapphire Rapids / Granite Rapids 有 AMX**（microarch.rs:125 `has_amx`）。AMD 无 AMX。AMX-FP16（TDPFP16PS）仅 Granite Rapids，AMX-COMPLEX/TRANSPOSE/FP8 是 Diamond Rapids（microarch.rs 预留 false）。

### 7. AVX10.2 + APX（下一代）

- **AVX10** 是统一 ISA 版本号，替代 AVX-512 的离散特性枚举：
  - **AVX10.1**（Version 1）= 枚举 AVX-512 特性，软件 enablement
  - **AVX10.2**（Version 2）= 新特性 + conformance level
- **Conformance level** 按向量宽度分级：**128-bit / 256-bit / 512-bit**（1024-bit 不在当前 spec）。N+1 是 N 的超集
- **APX（Advanced Performance Extensions）**：GPR 从 16 扩到 32（新增 r16-r30，实际可用 +15 / 总 31 个通用寄存器，r31 避开 RSP 语义），扩展寄存器文件需要更大 red-zone / spill 空间
- APX 通过 CPUID leaf 7 subleaf 1 检测（microarch.rs:204 `has_apx` 预留 false）

源码 `hardware_profile.rs:389` `gpr_count`：`CpuAvx10_2 => 31`，`CpuAvx512 | CpuAvx2 => 16`；`hardware_profile.rs:399` `num_simd_regs`：`CpuAvx512 | CpuAvx10_2 => 32`，`CpuAvx2 => 16`。

### 8. AVX-512 降频（license）— Zen4 double-pump

- **Skylake-X / Cascade Lake**：512-bit 指令触发重降频（heavy license），512-bit 吞吐可能不如 256-bit。microarch.rs:82 `zmm_downclocking()` 标记，`use_avx512()=false`（fallback 到 AVX2 geometry）
- **Zen4**：AVX-512 通过 **double-pump**（256-bit 单元分两拍执行 512-bit），无吞吐优势，`use_avx512()=false`，microkernel geometry 用 AVX2 的 (6,16,8)
- **Zen5 / Ice Lake / Tiger Lake / Sapphire Rapids / Granite Rapids**：native 512-bit，`use_avx512()=true`，geometry (14,32,16)

microarch.rs:46 `microkernel_geometry()` + :72 `use_avx512()` 是这两个决策的 SSOT。

---

## AI 易误判点（核心价值，堵幻觉）

| ❌ 误判 | ✅ 正解（官方 + 源码双确认） |
|--------|---------|
| W512 模式用 YMM 算子没问题（"YMM 是 ZMM 的低半别名，够用"）| YMM 写会清零 ZMM 高 256 位（官方"zero unused higher bits"规则）；YMM 读只能读低 8 lanes。W512 必须用 `if use_avx512 { ZMM 路径 }` 分流（BCE-20260703-AVX512-HALF-LANES 11 处根治） |
| `vextractf128` 能取 ZMM 高 lanes | 只能切 128-bit 边界，最大取 YMM 高 128（lanes 4-7）。取 ZMM 高 8 lanes 必须 `vextractf64x4(ymm_hi, zmm, 1)` |
| `VDPBF16PS` / `VPDPBUSD` 所有 AVX-512 CPU 都支持 | BF16/VNNI 是独立扩展（CPUID leaf7 subleaf1 EAX[5] / leaf7 ECX[11]）。Ice Lake/Tiger Lake 有 AVX-512 VNNI 但**无 BF16** → 只查 use_avx512 emit VDPBF16PS 会 SIGILL（BCE-20260704-X86-BF16-VNNI-GUARD） |
| `VCVTNEPS2BF16` 转换后 dst 是满的 | 只写 dst 低半，高半清零。VecNarrow 必须 vextracti128 分两路处理高半（BCE-20260708-VECNARROW-LANE-LOSS） |
| AMX 默认可用（CPU 有 AMX flag 就行）| 需 XCR0[18:17] + CR4.AMXFLAG + LDTILECFG(64-byte palette struct) 三步初始化，缺一会 `#UD`（BCE-20260703-CODEGEN-AUDIT AMX 子条目） |
| AMX tile 随便配尺寸 | tile 尺寸由 palette（0/1）枚举限定，LDTILECFG 的 palette_id 必须匹配硬件支持 |
| AVX-512 FP16 = F16C | 不同。F16C（Ivy Bridge+）只做 F16↔F32 转换；AVX-512 FP16（SPR+）做 F16 原生计算（microarch.rs:157 has_f16c vs :139 has_avx512fp16） |
| FP8 像 BF16/VNNI 有 AVX-512 硬件 dot | ❌ 不存在 AVX512_FP8 独立扩展。FP8 dot 仅 AMX-FP8（`TDPFP8PS`，Diamond Rapids 未上市）；AVX10.2 只有 FP8 转换指令（无原生 dot）。当前 gllm 所有支持的硬件 FP8 走 ScalarLUT 软件解码（`FP8 → DequantCompute(ScalarLUT)`，microarch `has_amx_fp8=false`） |
| Zen4 有 AVX-512 → use_avx512=true 用 512-bit | Zen4 double-pump（半速），无吞吐优势，microarch use_avx512=false，走 AVX2 geometry（microarch.rs:59） |
| zmm 32 寄存器随便分配，调用约定不管 | PhysVec 16..31 需 RegAllocator 显式覆盖；调用约定 caller/callee-saved 规则 + spill 规划（BCE-20260702-REGALLOC-AVX2-OOB：AVX2 RegAllocator 只覆盖 0-15，AVX-512 扩展寄存器 OOB） |
| BF16 权重要转 F32 喂下游 | 违宪（ARCH-BLOB-YIELDS-WEIGHT）。blob 保留 BF16 原始字节，JIT 在 SIMD 指令层 widen（`vpmovzxwd`+`vpslld`），见 `dtype-propagation.md` |

---

## 解决问题时参考（编码/诊断必对照）

### 半 lanes（AVX-512 W512 模式）
1. 任何 `lower_*_x86` 函数：先查有无 `if self.use_avx512` 分支；无 + 用 `scratch_ymm`/`ymmword_ptr`/`width.f32_lanes()` 的步长计算 = 半 lanes BUG（BCE-20260703-AVX512-HALF-LANES 同类判定）
2. ZMM reduce 三步：(a) `vextractf64x4(ymm_hi, zmm, 1)` 取高半 (b) op(ymm_lo, ymm_lo, ymm_hi) 16→8 (c) 8→4→2→1 xmm 链（参考 `zmm_hreduce_to_xmm` lower_instr.inc.rs:977）
3. VecStore/Broadcast 写 ZMM 后被 YMM-only 下游读 → 高 8 lanes 丢；反之 YMM-only 写 + ZMM 读 → 高 8 lanes 是垃圾/清零

### BF16/VNNI/FP16 指令 emit 前
- 每条扩展特性指令 emit 前必查对应 `has_*` flag（microarch.rs + hardware_profile.rs），**禁止只用 `use_avx512`**：
  - `vcvtneps2bf16` / `vdpbf16ps` → `has_bf16`
  - `vpdpbusd` → `has_vnni`
  - `vaddph` / `vfmadd213ph` → `has_avx512fp16`
  - AMX 指令 → `has_amx`（+ AMX+ 子特性 `has_amx_fp16` 等）
- `hardware_profile.rs:72-81`：`CpuAvx512`/`CpuAvx10_2` 类从 `MicroArch` 真实探测所有 `has_*`；`CpuAvx2` 全 false。`X86Lower` struct 必须携带 `has_bf16`/`has_vnni`/`has_avx512fp16` 字段

### VecNarrow（F32→BF16 store）
- `emit_f32_to_bf16_ymm_to_xmm_avx2`（emit_helpers.inc.rs:238）当前只取低半（`vpackusdw` 在 xmm 视图），高 4 lanes 丢。BF16 激活路径启用前必修：加 `vextracti128` 取高 4 lanes 完整 narrow 8 lanes
- 有 `vcvtneps2bf16` 的硬件（SPR+）：单指令只写 dst 低半，同样需注意 ZMM→YMM 收窄的高半

### AMX 启用序列
1. 检测：CPUID `has_amx`（仅 SPR/GNR）
2. XCR0[18:17] = 11b（OS 已设，应用层一般无需手动，但 AMX-aware 代码应验证）
3. 构造 64-byte TILECFG struct（palette_id + per-tile palette/rowsb/colsb/addr/stride）
4. `LDTILECFG [mem]` 加载配置
5. `TILELOAD` 装载 A/B tile（TMM0-7）
6. `TDPBF16PS tmm_dst, tmm_a, tmm_b`
7. `TILESTORE` 写回 + `TILERELEASE`（释放 tile 寄存器）

### 寄存器分配
- AVX-512 / AVX10.2：32 个 ZMM（PhysVec 0-31）。RegAllocator 必须覆盖 16..31（BCE-20260702-REGALLOC-AVX2-OOB）
- AVX2：16 个 YMM（PhysVec 0-15）
- 调用约定（SysV x86-64）：xmm/zmm0-15 caller-saved 的浮点参数；xmm/zmm16-31 caller-saved（AVX-512 新增）。Windows：xmm/zmm0-5 arg，部分 callee-saved

### CPUID 检测位速查（WikiChip §Detection 权威）
```
leaf 7 subleaf 0:
  EBX[16]=AVX512F  [17]=DQ  [28]=CD  [30]=BW  [31]=VL
  ECX[11]=VNNI     [14]=VPOPCNTDQ
  EDX[23]=FP16
leaf 7 subleaf 1:
  EAX[5]=BF16      EAX[7]=AVX-VNNI-INT8  EAX[23]=APX
leaf D subleaf 1（XCR0）:
  bit 17=TILECFG   bit 18=TILEDATA  (AMX state)
  bit 28=OPMASK (k0-7)  bit 7=Hi16_ZMM
```

---

## 已知问题 / 边界

### 降频与吞吐陷阱
- **Skylake-X / Cascade Lake**：ZMM 重降频（heavy license），512-bit 吞吐可能 < 256-bit。microarch `zmm_downclocking()=true`，`use_avx512()=false`
- **Zen4**：AVX-512 double-pump（256-bit 单元两拍），无吞吐优势。`use_avx512()=false`
- **Zen5 / SPR**：native 512-bit，无降频惩罚
- **AMD 无 AMX**：Zen4/Zen5 有 BF16+VNNI 但无 AMX；INT8/BF16 矩阵乘走 AVX-512 VNNI/VPDPBUSD 路径，不走 tile

### 半 lanes 的系统性根因
- gllm-kernels 有 **11 处已根治**的 YMM-only 算子（BCE-20260703-AVX512-HALF-LANES 7 处 + BCE-20260703-AVX512-HALF-LANES-2 4 处）：argmax/softmax_reduce_max/HReduce/Accumulate/softmax_normalize/temperature/Transcendental + ScaleApply/QuantBlockLoad Int8/VecUnaryOp/VecCmp
- 根治模式：`if self.use_avx512 { resolve_zmm + zmm 指令 (16 lanes) + spill_store_zmm } else { 原 YMM }`
- 新增 vec 算子 lowering 必须自带 `use_avx512` 分支（Code Review checklist）

### VmInstr::HReduce 无 width 字段
- `vminstr.inc.rs:242` `HReduce` 指令无 `width` 字段，lower 无法从指令知道 src 宽度，只能靠 `use_avx512` 推断。这是 HReduce 漏修半 lanes 的**结构原因**。新增水平归约类算子考虑在 VmInstr 携带 width 或显式走 ZMM 分支

### AMX palette 限制
- 当前 Sapphire Rapids 支持 palette 0/1（tile 尺寸组合有限）。palette 2（更大 tile / FP16）在 Granite Rapids+ 启用
- tile 寄存器仅 8 个（TMM0-7），每 tile 1KB，AMX-aware 代码需规划 tile 复用
- AMX 状态（TILECFG + TILEDATA）是独立的 XCR0 域，OS 必须显式启用（部分老内核默认关）

### AVX10.2 / APX 未落地
- microarch.rs:199-206 `has_avx10_2` / `has_apx` 预留 false（无 MicroArch 变体对应 Arrow Lake+ / Diamond Rapids）
- AVX10 conformance level（128/256/512）与 AVX-512 的关系：AVX10.1 = 枚举 AVX-512；AVX10.2 = 新特性。1024-bit 不在当前 spec
- APX 31 GPR 需新 ABI（red-zone 扩展 / spill 空间），JIT codegen 待适配

---

## 关键源码位置

| 关注点 | 文件:函数/行 |
|--------|------------|
| MicroArch 枚举 + 能力 flag | `gllm-kernels/src/microarch.rs:17` `MicroArch` enum + `has_amx`/`has_vnni`/`has_avx512fp16`/`has_bf16`/`has_f16c`/`has_avx10_2`/`has_apx` |
| use_avx512 + geometry 决策 | `microarch.rs:46` `microkernel_geometry()` + `:72` `use_avx512()` + `:82` `zmm_downclocking()` |
| HardwareProfile（CPU profile） | `gllm-kernels/src/compiler/hardware_profile.rs:12` `HardwareProfile` enum（`CpuAvx2`/`CpuAvx512`/`CpuAvx10_2`）+ `:59` `platform()` 从 MicroArch 探测 has_* + `:389` `gpr_count` + `:399` `num_simd_regs` |
| ZMM 高半提取（半 lanes 根治） | `gllm-kernels/src/compiler/codegen/vm/x86_lower/lower_instr.inc.rs:977` `zmm_hreduce_to_xmm`（`vextractf64x4 imm=1` + 8→4→2→1 链） |
| BF16 VNNI guard | `gllm-kernels/src/compiler/codegen/vm/x86_lower/lower_instr_dispatch.inc.rs` `has_bf16`/`has_vnni` 守卫 + `lower_dot_product_x86:1974` `has_avx512fp16` |
| F32→BF16 narrow（lane-loss） | `gllm-kernels/src/compiler/codegen/vm/x86_lower/emit_helpers.inc.rs:238` `emit_f32_to_bf16_ymm_to_xmm_avx2`（待加 vextracti128 高半） |
| AMX tile / palette | `gllm-kernels/src/compiler/codegen/vm/isa_profile.rs:45` 物理 Tile + `:487` `has_amx` feature + `:505` 8 tile_regs；`isa_hook.rs:214` `X86AmxPlusHook` |
| ISA feature 枚举 | `isa_profile.rs:487-494` `IsaFeature::Amx`/`AmxFp16`/`AmxComplex`/`AmxTranspose`/`AmxFp8` |
| AVX-512FP16 emit | `compiler/codegen/vm/instr_fragments/vminstr.inc.rs` + `op_impl.rs` + `isa_profile.rs` |

---

## 关联 BCE（16+ 同类，代码知识库交叉引用）

- **BCE-20260703-AVX512-HALF-LANES** — 7 处 reduction/scan 按 16-lane 步长跳但只处理低 8 lanes（SmolLM2 argmax=6）。根治 `zmm_hreduce_to_xmm`
- **BCE-20260703-AVX512-HALF-LANES-2** — 同类漏网 4 处（ScaleApply/QuantBlockLoad Int8/VecUnaryOp/VecCmp）
- **BCE-20260704-X86-BF16-VNNI-GUARD** — BF16/VNNI 指令只查 use_avx512 不查 has_bf16/has_vnni → 无 BF16/VNNI 的 AVX-512 CPU SIGILL（⚠️ 重开，fallback BF16 GEMM 数值对齐未验证）
- **BCE-20260702-REGALLOC-AVX2-OOB** — x86 RegAllocator YMM 范围 0-15 未覆盖 AVX-512 扩展寄存器 16-31
- **BCE-20260708-VECNARROW-LANE-LOSS** — `emit_f32_to_bf16_ymm_to_xmm_avx2` 丢高 4 lanes（次生 bug，BF16 激活启用时修）
- **BCE-20260703-CODEGEN-AUDIT** — 全设备 codegen 审计含 AMX 子条目
- **BCE-20260704-STEP-DTYPE-MISMATCH** — step 硬编码 F32 与 dtype elem_bytes 不匹配（NormLike stepbytes 同类，BF16 step=32 跳一半元素）

---

## 与其他资料库关系

- `dtype-propagation.md` — BF16/F32 在 x86 JIT 的 widen/narrow 传播链（本文件的 dtype 侧细节）
- `dot-product-cap-api.md` — `device.dot_product_cap()` 硬件能力 API（has_bf16/has_vnni/has_avx512fp16 的消费方）
- `kv-cache-dtype-dual-layer.md` — KV cache dtype 双地层陷阱（JIT ctx.dtype=F32 vs buffer compute_dtype=BF16）
- `BUG-KNOWLEDGE.md` — 上述 16+ BCE 完整归因（本文件是外部技术事实侧，BUG-KNOWLEDGE 是项目 BUG 模式侧，双沉淀交叉引用）
