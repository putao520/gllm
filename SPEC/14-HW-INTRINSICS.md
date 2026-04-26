# 硬件指令能力矩阵 (Hardware Instruction Intrinsics)

> **SSOT**: 本文档是 gllm-kernels JIT codegen 使用的硬件指令能力矩阵的唯一定义。描述每种硬件平台上可用的 GEMM/Attention/Norm/Dequant 指令、微核参数和寄存器约束。
>
> 交叉引用: `01-JIT-PIPELINE.md` §5 (StrategySelector)、`02-HARDWARE.md` (DeviceProfile)、`04-OPERATORS.md` (算子库)

## 1. GEMM 微核指令矩阵

### 1.1 x86_64 平台

#### 1.1.1 AVX2 VFMA GEMM

| 参数 | 值 |
|------|---|
| **指令** | `vfmadd231ps ymm, ymm, ymm` |
| **寄存器宽度** | 256-bit (ymm), 16 寄存器 |
| **微核尺寸** | MR=6, NR=4 (24 个 F32 元素/微核迭代) |
| **K 打包** | 4 路 FMA (K 维度每次处理 4 个元素) |
| **寄存器分配** | 6×4=24 累加器 + 4 读 A + 4 读 B = 32 (超出 16 ymm, 需分块重载) |
| **实际 MR×NR** | MR=6, NR=3 (受限于 16 ymm: 6×3=18 acc + 2 读 A + 1 读 B + 1 tmp = 22 spill, 实际 MR=4 NR=4 或 MR=6 NR=3) |
| **L2 分块** | KC ≤ L2/2, MC × KC ≤ L2 |
| **CPUID 检测** | `CPUID.07H:EBX.AVX2[5]=1` |

**微核伪代码** (MR=6, NR=4, 实际使用寄存器分块):

```asm
; 外层 KC 循环, 每次 4 个 F32
.vfma_loop:
    vbroadcastsd ymm_a0, [ptr_a + 0]   ; 加载 A 的 2 个 F32 广播
    vfmadd231ps ymm_c0, ymm_a0, ymm_b0  ; C[0:5][0:3] += A[0:5] * B[0:3]
    vfmadd231ps ymm_c1, ymm_a1, ymm_b1
    ; ... 继续下一个 K 块
```

#### 1.1.2 AVX-512 F32 GEMM

| 参数 | 值 |
|------|---|
| **指令** | `vfmadd231ps zmm, zmm, zmm` |
| **寄存器宽度** | 512-bit (zmm), 32 寄存器 |
| **微核尺寸** | MR=16, NR=6 (96 个 F32 元素/微核迭代) |
| **K 打包** | 4 路 FMA |
| **寄存器分配** | 16×6=96 累加器 zmm → 太多, 实际 MR×NR 受限: MR=12 NR=4 或 MR=8 NR=6 |
| **L2 分块** | KC ≤ L2/2, MC × KC ≤ L2 |
| **CPUID 检测** | `CPUID.07H:EBX.AVX512F[16]=1` |

#### 1.1.3 AVX-512 BF16 GEMM (VDPBF16PS)

| 参数 | 值 |
|------|---|
| **指令** | `vdpbf16ps zmm, zmm, zmm` |
| **功能** | 两对 BF16 值的点积 → F32 累加: `dst.f32[i] += src1.bf16[2i]*src2.bf16[2i] + src1.bf16[2i+1]*src2.bf16[2i+1]` |
| **寄存器宽度** | 512-bit (zmm), 32 寄存器 |
| **每指令吞吐** | 16 个 BF16 点积 → 16 个 F32 结果 (512/32=16) |
| **微核尺寸** | MR=16, NR=6 |
| **K 步进** | 每次 2 个 BF16 对 = 4 bytes (2×2byte BF16) |
| **输入格式** | A/B 矩阵保持 BF16 格式 (2 bytes/element) — 节省 2× 内存带宽 |
| **CPUID 检测** | `CPUID.07H.01H:EDX.AVX512_BF16[22]=1` ( Sapphire Rapids+) |

**融合 dequant 场景**: 权重保持 BF16 → 直接用 VDPBF16PS 计算, 无需先转 F32。对于 Q4 权重, 需要先解包到 BF16/F32 再计算。

**微核伪代码**:

```asm
; A 矩阵 BF16 打包, B 矩阵 BF16 打包
.vdpbf16ps_loop:
    vmovdqu16 zmm_a, [ptr_a + k*32]    ; 加载 16×2 = 32 个 BF16
    vmovdqu16 zmm_b, [ptr_b + k*32]    ; 加载 2×16 = 32 个 BF16
    vdpbf16ps zmm_c0, zmm_a, zmm_b     ; 16 个 BF16 点积 → F32 累加
    vdpbf16ps zmm_c1, zmm_a, zmm_b_next
    ; ...
```

#### 1.1.4 Intel AMX (Advanced Matrix Extensions)

| 参数 | 值 |
|------|---|
| **指令** | `TDPBF16PS tmm_dst, tmm_a, tmm_b` |
| **功能** | Tile 级 BF16 点积 → F32 累加: `dst[i][j] += Σk srcA.bf16[i][2k]*srcB.bf16[k][2j] + srcA.bf16[i][2k+1]*srcB.bf16[k][2j+1]` |
| **Tile 尺寸** | 最大 16 行 × 64 字节/行 (16×16 F32 或 16×32 BF16) |
| **Tile 数量** | 8 个 tile (TMM0-TMM7) |
| **PALETTE** | palette_id=1 (当前唯一支持) |
| **总存储** | 8 tiles × 16 rows × 64 bytes = 8 KB |
| **CPUID 检测** | `CPUID.07H.01H:EDX.AMX_BF16[22]=1`, `CPUID.07H.01H:EAX.AMX_TILE[24]=1` |

**Tile 配置结构 (TILECFG, 64 bytes)**:

| 偏移 | 大小 | 字段 |
|------|------|------|
| 0 | 1B | `palette_id` (必须 = 1) |
| 1 | 1B | `start_row` |
| 16-23 | 8×1B | `colsb[0..7]` — 每 tile 每行字节数 |
| 32-39 | 8×1B | `rows[0..7]` — 每 tile 行数 |

**相关 AMX 指令**:

| 指令 | 功能 |
|------|------|
| `LDTILECFG [mem]` | 从内存加载 tile 配置 |
| `STTILECFG [mem]` | 保存 tile 配置到内存 |
| `TILELOADD tmm, [mem]` | 从内存加载 tile 数据 |
| `TILESTORED [mem], tmm` | 存储 tile 数据到内存 |
| `TILEZERO tmm` | 清零 tile |
| `TILERELEASE` | 释放 tile 配置 |
| `TDPBF16PS tmm1, tmm2, tmm3` | BF16×BF16 → F32 累加 |
| `TDPBSSD tmm1, tmm2, tmm3` | INT8×INT8 → INT32 累加 |
| `TDPBUUD tmm1, tmm2, tmm3` | UINT8×UINT8 → INT32 累加 |

**GEMM 分块策略**: AMX tile 最大 16×16 F32, 所以 GEMM 需要按 16 分块:

```
for mc in (0..M).step_by(16):
    for nc in (0..N).step_by(16):
        TILELOADD tmm_c ← C[mc..mc+16, nc..nc+16]
        for kc in (0..K).step_by(32):  // BF16: 2 per pair
            TILELOADD tmm_a ← A[mc..mc+16, kc..kc+32]
            TILELOADD tmm_b ← B[kc..kc+32, nc..nc+16]
            TDPBF16PS tmm_c, tmm_a, tmm_b
        TILESTORED C[mc..mc+16, nc..nc+16] ← tmm_c
```

### 1.2 AArch64 平台

#### 1.2.1 NEON FMLA GEMM

| 参数 | 值 |
|------|---|
| **指令** | `fmla v0.4s, v1.4s, v2.4s` |
| **寄存器宽度** | 128-bit (v0-v31, 32 寄存器) |
| **每指令** | 4 个 F32 FMA |
| **微核尺寸** | MR=8, NR=4 (32 个 F32 元素/微核迭代) |
| **寄存器分配** | 8×4=32 累加器 (刚好用完 32 个 v 寄存器, 需要节省) → 实际 MR=8 NR=3 |
| **CPUID 检测** | `ID_AA64PFR0_EL1.AdvSIMD[20:23] != 0` (所有 ARMv8+) |

#### 1.2.2 SVE FMLA GEMM

| 参数 | 值 |
|------|---|
| **指令** | `fmla z0.s, p0/m, z1.s, z2.s` |
| **寄存器宽度** | 128~2048-bit (z0-z31), 可变长度 (VL) |
| **每指令** | VL/32 个 F32 FMA (256-bit VL = 8, 512-bit VL = 16) |
| **微核尺寸** | MR = VL/32 × 2, NR = 4 (自适应 VL) |
| **Predicate** | `p0-p15` 掩码寄存器, 用于尾部处理 |
| **CPUID 检测** | `ID_AA64PFR0_EL1.SVE[32:35] != 0` |

#### 1.2.3 SVE2 FMMLA (矩阵乘累加)

| 参数 | 值 |
|------|---|
| **指令** | `FMMLA Zda.S, Zn.S, Zm.S` |
| **功能** | 外积: `Zda[i][j] += Zn[i] * Zm[j]` (2D 矩阵累加) |
| **输入精度** | FP32 (`.S`) |
| **累加精度** | FP32 |
| **矩阵尺寸** | 取决于 VL: 256-bit → 2×4 外积块; 512-bit → 4×4 |
| **每指令吞吐** | 2×VL/32 × VL/32 个 FMA (远超逐元素 FMLA) |
| **CPUID 检测** | `ID_AA64PFR0_EL1.SVE[32:35] >= 1` + SVE2 矩阵扩展 |

#### 1.2.4 SVE2 BFMMLA (BF16 矩阵乘累加)

| 参数 | 值 |
|------|---|
| **指令** | `BFMMLA Zda.S, Zn.H, Zm.H` |
| **功能** | BF16 外积 → F32 累加: `Zda.f32[i][j] += bf16_to_f32(Zn.bf16[2i]) * bf16_to_f32(Zm.bf16[2j]) + bf16_to_f32(Zn.bf16[2i+1]) * bf16_to_f32(Zm.bf16[2j+1])` |
| **输入精度** | BF16 (`.H`) |
| **累加精度** | FP32 (`.S`) |
| **吞吐优势** | 输入 16-bit, 吞吐是 FMMLA 的 ~2× (相同带宽下 2× 元素) |
| **CPUID 检测** | `ID_AA64PFR0_EL1.SVE[32:35] >= 1` + `ID_AA64PFR0_EL1.BF16[44:47] != 0` |

### 1.3 NVIDIA GPU 平台

#### 1.3.1 SM70 (Volta) — WMMA

| 参数 | 值 |
|------|---|
| **指令** | `wmma.mma.sync.aligned.m16n16k16.f32.tf32.tf32.f32` (或 `.f16.f16.f32`) |
| **矩阵尺寸** | 16×16×16 (M×N×K) per WMMA |
| **输入精度** | FP16 (`.f16`) |
| **累加精度** | FP32 (`.f32`) |
| **线程模型** | 单 warp (32 threads) 协作 |
| **Stage** | 通常 2 stage (double buffer) |
| **Shared Memory** | 手动管理 (无 TMA) |
| **检测** | `cudaGetDeviceProperties().major == 7, .minor == 0` |

#### 1.3.2 SM80 (Ampere) — MMA.sync + cp.async

| 参数 | 值 |
|------|---|
| **指令** | `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` (或 `.f16.f16.f32`) |
| **矩阵尺寸** | 16×8×16 (M×N×K) per MMA (可组合为更大 tile) |
| **输入精度** | BF16 / FP16 / TF32 / INT8 |
| **累加精度** | FP32 / INT32 |
| **异步拷贝** | `cp.async.ca.shared.global [smem], [gmem], 16` — 异步从全局内存到 shared memory |
| **Stage** | 通常 3-4 stage (多缓冲 + cp.async 流水线) |
| **线程模型** | 单 warp (32 threads) 协作 |
| **检测** | `cudaGetDeviceProperties().major == 8, .minor == 0/6/9` |

**多 stage 流水线**:

```
Stage 0 (计算): MMA.sync(C_tile, A_tile[cur], B_tile[cur])
Stage 1 (加载): cp.async(A_tile[next], A_global[next_k])
Stage 2 (加载): cp.async(B_tile[next], B_global[next_k])
```

**INT8 IMMA (整数量化 GEMM)**:

| 参数 | 值 |
|------|---|
| **指令** | `mma.sync.aligned.m16n8k32.s32.u8.u8.s32` |
| **输入** | UINT8 × UINT8 |
| **累加** | INT32 |
| **适用** | INT8 量化权重 + INT8 激活 |

#### 1.3.3 SM90 (Hopper) — WGMMA + TMA

| 参数 | 值 |
|------|---|
| **指令** | `wgmma.mma_async.sync.aligned.m64nNkK.f32.bf16.bf16` |
| **矩阵尺寸** | M=64 (固定), N=8/16/24/.../256, K=16(BF16)/32(FP8) |
| **输入精度** | BF16 / FP16 / TF32 / FP8(E4M3/E5M2) / INT8 |
| **累加精度** | FP32 / INT32 |
| **Warp Group** | 4 个 warp = 128 threads 协作 (非单 warp) |
| **TMA** | `cp.async.bulk.tensor` — 硬件加速多维 tensor 拷贝, 单线程发起 |
| **Cluster** | Thread Block Cluster (分布式 shared memory) |
| **多 Wave** | `num_waves` 个 warp group 协作同一 tile |
| **检测** | `cudaGetDeviceProperties().major == 9, .minor == 0` |

**TMA (Tensor Memory Accelerator)**:

| 指令 | 功能 |
|------|------|
| `cp.async.bulk.tensor.2d.shared.global` | 2D tensor 异步拷贝到 shared memory |
| `cp.async.bulk.tensor.3d.shared.global` | 3D tensor 异步拷贝 |
| `cp.async.bulk.commit_group` | 提交一组异步操作 |
| `cp.async.bulk.wait_group N` | 等待最近 N 组完成 |
| `bar.cluster` | Cluster 级 barrier (分布式 shared memory 同步) |

**WGMMA fence/wait**:

| 指令 | 功能 |
|------|------|
| `wgmma.fence` | 确保 WGMMA 操作数对 warp group 可见 |
| `wgmma.commit_group` | 提交一组 WGMMA 操作 |
| `wgmma.wait_group N` | 等待最近 N 组 WGMMA 完成 |

**FP8 Tensor Core**:

| 参数 | 值 |
|------|---|
| **指令** | `wgmma.mma_async...f32.e4m3.e4m3` / `.e5m2.e5m2` |
| **输入** | FP8 E4M3 或 E5M2 |
| **累加** | FP32 |
| **K 步进** | 32 (vs BF16 的 16) — 2× 吞吐 |

#### 1.3.4 SM100+ (Blackwell) — tcgen05

| 参数 | 值 |
|------|---|
| **指令** | `tcgen05.mma...` (5th-gen tensor core) |
| **矩阵尺寸** | 更大 tile (具体尺寸待公开) |
| **输入精度** | BF16 / FP8(E4M3/E5M2) / INT8 / FP4 (新增) |
| **累加精度** | FP32 / INT32 |
| **TMEM** | Tensor Memory — 片上 tensor 专用存储, 替代寄存器累加 |
| **检测** | `cudaGetDeviceProperties().major == 10` (预计) |

**新增能力**:
- FP4 输入支持 (2-bit 浮点) — 4× 内存节省 vs FP8
- TMEM (Tensor Memory) — 专用累加器存储, 减少寄存器压力
- 更大的单指令 tile 尺寸

### 1.4 AMD GPU 平台

#### 1.4.1 RDNA 3 (GFX11) — WMMA

| 参数 | 值 |
|------|---|
| **指令** | `v_wmma_f32_16x16x16_f16` / `v_wmma_f32_16x16x16_bf16` |
| **矩阵尺寸** | 16×16×16 (M×N×K) per WMMA |
| **输入精度** | FP16 / BF16 |
| **累加精度** | FP32 |
| **线程模型** | Wavefront (32 lanes) 协作 |
| **检测** | AMD GPU architecture `gfx1100`/`gfx1101`/`gfx1102` |

#### 1.4.2 RDNA 4 (GFX12) — WMMA 扩展

| 参数 | 值 |
|------|---|
| **F16 路径** | `v_wmma_f32_16x16x16_f16` |
| **BF16 路径** | `v_wmma_f32_16x16x16_bf16` |
| **a4w8 路径** | `v_wmma_i32_16x16x16_iu4_iu8` (混合精度: 4-bit × 8-bit) |
| **INT8 路径** | `v_wmma_i32_16x16x16_iu8` |
| **矩阵尺寸** | 16×16×16 per WMMA |
| **累加** | FP32 (浮点路径) / INT32 (整数路径) |
| **线程模型** | Wavefront (32 lanes) 协作 |
| **检测** | AMD GPU architecture `gfx1200`/`gfx1201` |

**a4w8 混合精度 WMMA 详解**:

```
数学语义:  C[f32] += dequant(A[4-bit], scale_a) × dequant(B[8-bit], scale_b)

硬件行为:
  - A 矩阵: 4-bit 整数 (packed, 每 32-bit 寄存器存 8 个值)
  - B 矩阵: 8-bit 整数 (每 32-bit 寄存器存 4 个值)
  - 累加: INT32 (可选转为 FP32)
  - Scale/dequant: 硬件隐式完成, 零软件开销

适用场景:
  - 权重保持 Q8_0 格式 → 直接输入 WMMA
  - 激活保持 Q4_0 格式 → 直接输入 WMMA
  - 替代软件 dequant+GEMM, 节省内存带宽 4-8×

约束:
  - Block size 必须对齐到 16 (WMMA tile 维度)
  - Scale/zp 需要 extra register 存放 (per-block)
```

## 2. Attention 微核指令矩阵

### 2.1 CPU Tiled Attention

Attention 不像 GEMM 有硬件矩阵指令，而是通过 cache-blocking 分块 + SIMD 向量化实现:

| 平台 | QK^T 计算 | Softmax | AV × V | 分块策略 |
|------|----------|---------|--------|---------|
| AVX2 | `vmulps` + `vfmadd231ps` 归约 | `vexp_ps` 逼近 + `vdivps` | `vfmadd231ps` | tile_q=64, tile_k=64 |
| AVX-512 | `vmulps zmm` + `vfmadd231ps zmm` 归约 | `vexp_ps zmm` 逼近 + `vdivps zmm` | `vfmadd231ps zmm` | tile_q=128, tile_k=128 |
| NEON | `fmul` + `fmla` 归约 | 多项式 exp + `fdiv` | `fmla` | tile_q=64, tile_k=64 |
| SVE | `fmul z` + `fmla z` 归约 | 多项式 exp + `fdiv z` | `fmla z` | tile_q=VL*4, tile_k=VL*4 |

**Softmax 安全计算 (避免数值溢出)**:

```
max_val = reduce_max(scores)           // 向量化 max 归约
shifted = scores - max_val             // 向量化减法
exp_vals = approx_exp(shifted)         // 多项式逼近 (6-12 条 SIMD 指令)
sum_exp = reduce_sum(exp_vals)         // 向量化 add 归约
prob = exp_vals / sum_exp              // 向量化除法
```

### 2.2 GPU Flash Attention

| 世代 | Shared Memory | Attention Tile | Softmax | KV 读取 |
|------|--------------|---------------|---------|--------|
| SM80 (Flash V1) | 手动 cp.async | block_q=64, block_k=64 | online softmax (max+sum) | 分 stage 异步预取 |
| SM90 (Flash V2) | TMA bulk copy | block_q=128, block_k=128 | online softmax + swizzle | TMA 单线程发起 |

**Online Softmax (Flash Attention 核心)**:

```
// 不需要完整 QK^T 矩阵, 逐 K 块流式计算:
for each K block:
    scores = Q * K_block^T       // MMA
    max_new = max(max_old, max(scores))
    correction = exp(max_old - max_new)
    sum_new = sum_old * correction + sum(exp(scores - max_new))
    O = O * correction + exp(scores - max_new) * V_block  // MMA
```

## 3. Fused Dequant 指令矩阵

### 3.1 Q4 解包 → FMA 融合

**场景**: Q4_K 权重保持 4-bit 压缩格式, GEMM 微核内循环先解包再乘累加

```
// 微核内 K 循环 (每次处理 block_size=32 个 Q4 元素)
for k_block in (0..K).step_by(block_size):
    // 1. 加载 Q4 packed bytes (block_size/2 = 16 bytes)
    load q4_packed ← weight_ptr[k_block/2]

    // 2. 解包 4-bit → F32 (SIMD)
    //    x86: vpslld + vpsrld 移位拆分高低 4-bit
    //    ARM: shrn + sli 移位拆分
    unpack_lo = (q4_packed & 0x0F) - zp   // 低 4-bit → int
    unpack_hi = (q4_packed >> 4) - zp     // 高 4-bit → int
    f32_vals = (unpack_lo + unpack_hi) * scale  // int → f32

    // 3. FMA 累加 (与标准 GEMM 微核相同)
    C[mr][nr] += f32_vals * activation[k_block]
```

**额外开销 vs 纯 F32 GEMM**: 约 3-5 条额外 SIMD 指令/K 迭代 (解包 + scale 乘)

### 3.2 Q8 解包 → FMA 融合

```
// 微核内 K 循环 (每次处理 block_size 个 Q8 元素)
for k_block in (0..K).step_by(block_size):
    // 1. 加载 Q8 bytes + scale
    load q8_bytes ← weight_ptr[k_block]     // block_size 个 INT8
    load scale    ← scale_ptr[block_idx]    // 1 个 F32 scale

    // 2. INT8 → F32 + scale
    //    x86: vpmovsxbd + vcvtdq2ps + vmulps
    //    ARM: sshll + scvtf + fmul
    f32_vals = int32(q8_bytes) * scale

    // 3. FMA 累加
    C[mr][nr] += f32_vals * activation[k_block]
```

**额外开销**: 约 2-3 条额外 SIMD 指令/K 迭代 (INT8→F32 转换 + scale 乘)

### 3.3 MXFP4 双指针读取

**场景**: MXFP4 格式 blocks 和 scales 存储在相邻但独立的内存区域 (ARCH-MXFP4-SEPARATE)

```
// 双指针: blocks_ptr 指向 4-bit blocks, scales_ptr 指向 F8 E8M0 scales
for k_block in (0..K).step_by(block_size=32):
    // 1. 加载 blocks (32 个 4-bit = 16 bytes)
    load mxfp4_blocks ← blocks_ptr[k_block/2]

    // 2. 加载 scale (1 个 E8M0 = 1 byte per block)
    load e8m0_scale ← scales_ptr[block_idx]

    // 3. 解码: 4-bit → F32 × E8M0 scale
    //    E8M0 纯指数格式: value = 2^(exponent - 127)
    //    4-bit mantissa → F32 × scale
    f32_vals = decode_mxfp4(mxfp4_blocks, e8m0_scale)

    // 4. FMA 累加
    C[mr][nr] += f32_vals * activation[k_block]
```

**额外开销**: 约 5-8 条额外 SIMD 指令/K 迭代 (4-bit 解包 + E8M0 解码 + 乘法)

### 3.4 硬件隐式 Dequant

| 硬件 | 指令 | 输入格式 | 累加格式 | 额外软件开销 |
|------|------|---------|---------|-------------|
| AMD GFX12 | `v_wmma_i32_16x16x16_iu4_iu8` | Q4 × Q8 | INT32→FP32 | 0 — 硬件完成 |
| AMD CDNA 3 (gfx942) | `v_mfma_i32_32x32x32i8` | INT8 × INT8 | INT32→FP32 | 0 — 硬件完成 |
| AMD CDNA 3 (gfx942) | `v_mfma_f32_32x32x32_bf8_bf8` | FP8 BF8 × BF8 | FP32 | 0 — 硬件完成 |
| NVIDIA SM80 | `mma.sync...s32.u8.u8.s32` | INT8 × INT8 | INT32→FP32 | 0 — 硬件完成 |
| NVIDIA SM90 | `wgmma...f32.e4m3.e4m3` | FP8 × FP8 | FP32 | 0 — 硬件完成 |
| Intel AMX | `TDPBSSD` | INT8 × INT8 | INT32→FP32 | 0 — 硬件完成 |

**关键**: 硬件隐式 dequant 路径的额外开销为零。JIT 只需要将压缩格式的数据直接喂给硬件 WMMA/MMA 指令, 不需要任何软件解包步骤。这是 compute-time dequant 的最优路径。

## 4. 向量化 Elementwise / Norm 指令矩阵

### 4.1 Elementwise 基本指令

| 操作 | x86 AVX2 | x86 AVX-512 | ARM NEON | ARM SVE |
|------|---------|------------|----------|---------|
| Add | `vaddps ymm` | `vaddps zmm` | `fadd v.4s` | `fadd z.s` |
| Mul | `vmulps ymm` | `vmulps zmm` | `fmul v.4s` | `fmul z.s` |
| FMA | `vfmadd231ps ymm` | `vfmadd231ps zmm` | `fmla v.4s` | `fmla z.s p0/m` |
| Max | `vmaxps ymm` | `vmaxps zmm` | `fmax v.4s` | `fmax z.s` |
| Min | `vminps ymm` | `vminps zmm` | `fmin v.4s` | `fmin z.s` |
| Sqrt | `vsqrtps ymm` | `vsqrtps zmm` | `fsqrt v.4s` | `fsqrt z.s` |
| Rcp | `vrcpps ymm` | `vrcpps zmm` | `frecpe v.4s` | `frecpe z.s` |

### 4.2 Transcendental (多项式逼近)

| 操作 | x86 指令数 | ARM 指令数 | 精度 |
|------|----------|----------|------|
| Exp | ~12 条 (range reduce + Horner) | ~10 条 | ULP < 2 |
| Tanh | ~16 条 (exp+x / exp-x) | ~14 条 | ULP < 4 |
| Log | ~10 条 (range reduce + Horner) | ~8 条 | ULP < 2 |
| Sigmoid | ~14 条 (exp + div) | ~12 条 | ULP < 4 |

### 4.3 Norm (RmsNorm / LayerNorm)

Norm 操作 = 两遍扫描 + scale:

```
// Pass 1: 归约 (向量化 reduce)
sum_sq = reduce_add(vec_x * vec_x)  // vfmadd231ps / fmla 归约
inv_rms = 1.0 / sqrt(sum_sq / n + eps)

// Pass 2: 缩放 (向量化 elementwise)
for i in 0..n:
    out[i] = x[i] * inv_rms * weight[i]  // vmulps × 2 / fmul × 2
```

| 归约操作 | x86 AVX2 | x86 AVX-512 | ARM NEON | ARM SVE |
|---------|---------|------------|----------|---------|
| Reduce-Add | horizontal add (4 步) | `vreduceps` (1 步) | pairwise add (3 步) | `faddv` (1 步) |
| Reduce-Max | horizontal max (4 步) | `vreducemaxps` (1 步) | pairwise max (3 步) | `fmaxv` (1 步) |

## 5. 设备能力探测方法

### 5.1 x86_64 CPUID 探测

| 能力 | CPUID 叶片 | 位 | 对应 DeviceProfile 字段 |
|------|----------|---|----------------------|
| AVX2 | `07H:EBX` | bit 5 | `has_avx2` |
| AVX-512 F | `07H:EBX` | bit 16 | `has_avx512` |
| AVX-512 BF16 | `07H.01H:EDX` | bit 22 | `has_avx512_bf16` |
| AVX-512 VNNI | `07H:ECX` | bit 11 | `has_avx512_vnni` |
| AMX TILE | `07H.01H:EAX` | bit 24 | `has_amx` |
| AMX BF16 | `07H.01H:EDX` | bit 22 | `has_amx_bf16` |
| AMX INT8 | `07H.01H:EDX` | bit 25 | `has_amx_int8` |
| FPU GFNI | `07H:ECX` | bit 14 | (间接, 用于 bit manipulation) |

**cache 层级探测**: `CPUID.04H` (L1/L2), `CPUID.8000001DH` (AMD L3)

### 5.2 AArch64 系统寄存器探测

| 能力 | 系统寄存器 | 位 | 对应 DeviceProfile 字段 |
|------|----------|---|----------------------|
| NEON | `ID_AA64PFR0_EL1[20:23]` | ≠ 0 | `has_neon` |
| SVE | `ID_AA64PFR0_EL1[32:35]` | ≠ 0 | `has_sve` |
| SVE VL | `RDVL` 指令 | — | `sve_vl_bytes` |
| BF16 | `ID_AA64PFR0_EL1[44:47]` | ≠ 0 | `has_bf16` |
| FMMLA | SVE2 矩阵扩展 | — | `has_fmmla` |
| I8MM | `ID_AA64PFR0_EL1[52:55]` | ≠ 0 | `has_i8mm` (INT8 矩阵乘) |

### 5.3 NVIDIA GPU 属性探测

| 能力 | CUDA API | 对应 DeviceProfile 字段 |
|------|---------|----------------------|
| SM 版本 | `cudaGetDeviceProperties().major*10+minor` | `sm_version` |
| Tensor Core 代数 | 从 SM 版本推导 | `tensor_core_gen` |
| Shared Memory / Block | `sharedMemPerBlock` | `smem_per_block` |
| Shared Memory / SM | `sharedMemPerMultiprocessor` | `smem_per_sm` |
| Warp Size | `warpSize` (固定 32) | `warp_size` |
| Max Threads / Block | `maxThreadsPerBlock` | `max_threads_per_block` |
| L2 Cache | `l2CacheSize` | `l2_cache_bytes` |
| Memory Bandwidth | `memoryBusWidth` + clock | `mem_bandwidth_gb_s` |

**Tensor Core 代数推导**:

| SM 版本 | 代数 | 指令接口 | FP16 | BF16 | FP8 | INT8 |
|--------|------|---------|------|------|-----|------|
| 70 | V1 | WMMA | ✅ | ❌ | ❌ | ❌ |
| 75 | V2 | WMMA | ✅ | ❌ | ❌ | ❌ |
| 80 | V3 | MMA.sync | ✅ | ✅ | ❌ | ✅ |
| 86 | V3 | MMA.sync | ✅ | ✅ | ❌ | ✅ |
| 89 | V3 | MMA.sync | ✅ | ✅ | ❌ | ✅ |
| 90 | V4 | WGMMA | ✅ | ✅ | ✅ | ✅ |
| 100+ | V5 | tcgen05 | ✅ | ✅ | ✅ | ✅ |

### 5.4 AMD GPU 属性探测

| 能力 | ROCm API | 对应 DeviceProfile 字段 |
|------|---------|----------------------|
| Architecture | `hipGetDeviceProperties().gcnArchName` | `gfx_arch` ("gfx1100", "gfx1200"...) |
| Wavefront Size | `warpSize` (固定 32 / 64 RDNA=32, CDNA=64) | `wavefront_size` |
| Shared Memory / Block | `sharedMemPerBlock` | `smem_per_block` |
| L2 Cache | `l2CacheSize` | `l2_cache_bytes` |
| WMMA 支持 | 从 `gcnArch` 推导 | `has_wmma`, `wmma_formats` |
| MFMA 支持 | 从 `gcnArch` 推导 (gfx90a/gfx942) | `has_mfma`, `mfma_formats` |
| MFMA INT8 | 从 `gcnArch` 推导 (gfx942+) | `has_mfma_int8` |
| MFMA FP8 | 从 `gcnArch` 推导 (gfx942+) | `has_mfma_fp8` |

**AMD GPU 架构矩阵**:

| gfx 架构 | 产品代号 | Wavefront | WMMA | MFMA | BF16 | FP8 | a4w8 | INT8 MFMA | FP8 MFMA |
|---------|---------|-----------|------|------|------|-----|------|-----------|-----------|
| gfx1100 | RDNA 3 (Navi 31) | 32 | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| gfx1101 | RDNA 3 (Navi 32) | 32 | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| gfx1102 | RDNA 3 (Navi 33) | 32 | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| gfx1200 | RDNA 4 (Navi 48) | 32 | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| gfx1201 | RDNA 4 (Navi 44) | 32 | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| gfx90a | CDNA 2 (MI250/MI210) | 64 | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| gfx942 | CDNA 3 (MI300X/MI300A) | 64 | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |

## 6. 策略选择→指令发射映射总结

完整映射链: `ComputeStrategy variant → 具体硬件指令 → DeviceProfile 检测字段`

| ComputeStrategy | 核心指令 | DeviceProfile 检测 |
|----------------|---------|-------------------|
| `ScalarLoop` | 标量三重循环 | (无硬件要求) |
| `Avx2Fma` | `vfmadd231ps ymm` | `has_avx2` |
| `Avx512F32` | `vfmadd231ps zmm` | `has_avx512` |
| `Avx512Bf16` | `vdpbf16ps zmm` | `has_avx512_bf16` |
| `AmxTile` | `TDPBF16PS tmm` | `has_amx && has_amx_bf16` |
| `NeonFmla` | `fmla v.4s` | `has_neon` |
| `SveFmla` | `fmla z.s p0/m` | `has_sve` |
| `Sm70Wmma` | `wmma.mma.sync...f16` | `sm_version == 70` |
| `Sm80MmaAsync` | `mma.sync...bf16` + `cp.async` | `sm_version == 80` |
| `Sm90WgmmaTma` | `wgmma.mma_async...bf16` + `cp.async.bulk.tensor` | `sm_version == 90` |
| `Sm100Tcgen05` | `tcgen05.mma...` | `sm_version >= 100` |
| `Rdna3Wmma` | `v_wmma_f32_16x16x16_f16` | `gfx_arch == "gfx11*"` |
| `Rdna4WmmaF16` | `v_wmma_f32_16x16x16_f16` | `gfx_arch == "gfx12*"` |
| `Rdna4WmmaBf16` | `v_wmma_f32_16x16x16_bf16` | `gfx_arch == "gfx12*"` |
| `Rdna4WmmaA4W8` | `v_wmma_i32_16x16x16_iu4_iu8` | `gfx_arch == "gfx12*" && has_wmma_a4w8` |
| `FusedQ4Gemm` | 解包(VPSLLD+VPSRLD) + `vfmadd231ps` | `has_avx2` (或等价) |
| `FusedQ8Gemm` | 转换(VPMOVSXBD+VCVTDQ2PS) + `vfmadd231ps` | `has_avx2` (或等价) |
| `FusedMxfp4Gemm` | 双指针加载 + 解码 + `vfmadd231ps` | `has_avx2` (或等价) |

## 7. 参考文档

| 平台 | 文档 |
|------|------|
| Intel x86 | [Intel 64 and IA-32 Architectures Software Developer's Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html) |
| Intel AMX | [Intel AMX Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.htm#techs=AMX) |
| ARM SVE | [ARM Architecture Reference Manual ARMv9-A](https://developer.arm.com/documentation/ddi0487/latest) |
| NVIDIA PTX | [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/) |
| NVIDIA CUTLASS | [CUTLASS 3.x GEMM/Kernel Library](https://github.com/NVIDIA/cutlass) |
| AMD GPUOpen | [AMD GPUOpen ISA Documentation](https://github.com/GPUOpen-Tools/llpc) |
| AMD ROCm | [ROCm Documentation](https://rocm.docs.amd.com/) |
