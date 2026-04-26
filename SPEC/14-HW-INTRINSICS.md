# 硬件指令能力矩阵

> **SSOT**: 本文档定义 gllm-kernels JIT codegen 目标的所有硬件指令序列、微核参数和设备探测方法。StrategySelector (`01-JIT-PIPELINE.md` §5) 根据 `DeviceProfile` (`02-HARDWARE.md`) 选择 ComputeStrategy 变体后，Phase 3 codegen 按本文档的映射规则发射机器码。

## 1. GEMM 微核指令矩阵

### 1.1 x86_64 GEMM

| 变体 | ISA | 微核尺寸 MR×NR | 累加器寄存器 | 关键指令 | 适用条件 |
|------|-----|---------------|------------|---------|---------|
| `ScalarLoop` | 基准 | 1×1 | 1 | `mulss` + `addss` | 无 SIMD (fallback) |
| `Avx2Fma` | AVX2 | 6×4 | 24 ymm | `vfmadd231ps ymm` | 16 ymm regs, 6×4 + pack |
| `Avx512F32` | AVX-512 (VNNI 无) | 16×6 | 96 zmm | `vfmadd231ps zmm` | 32 zmm regs |
| `Avx512Bf16` | AVX-512 + VDPBF16PS | 16×6 | 96 zmm | `vdpbf16ps zmm` | BF16 输入, F32 累加 |
| `AmxTile` | AMX (TILECONFIG) | 16×16 tile | TMEM | `tdpbf16ps tmm` | Intel Xeon Scalable |

**Avx2Fma 微核结构** (MR=6, NR=4):
```
pack_B: [KC, NR] → [NR, KC] (转置+对齐)
pack_A: [MC, KC] → row-major

for j in 0..NC step NR:
    pack_B_panel(j)
    for i in 0..MC step MR:
        load_C(MR×NR) → ymm[0..23]  // 6×4=24 累加器
        for p in 0..KC:
            broadcast A[i+p] → ymm_a
            load B[p, j..j+NR] → ymm_b
            vfmadd231ps ymm_acc, ymm_a, ymm_b
        store_C(ymm[0..23] → C[i..i+MR, j..j+NR])
```

**寄存器预算** (Avx2Fma): 16 ymm 总 — 24 用于累加器（分时复用）+ 2 pack_a + 2 pack_b + 1 broadcast + 1 temp。MR×NR 不得超过可用寄存器数。

**Avx512Bf16 微核**: 与 Avx512F32 结构相同，但输入为 BF16 pair，`vdpbf16ps` 一条指令完成两个 BF16→F32 乘加。有效吞吐翻倍。

**AmxTile 微核**: 使用 8 个 tile 寄存器 (TMM0-TMM7)。`tdpbf16ps` 一条指令完成 16×16×16 BF16 GEMM tile 累加。需要 `tileload` / `tilestore` 配合。

### 1.2 AArch64 GEMM

| 变体 | ISA | 微核尺寸 MR×NR | 关键指令 | 适用条件 |
|------|-----|---------------|---------|---------|
| `ScalarLoop` | 基准 | 1×1 | `fmadds` | 无 NEON (fallback) |
| `NeonFmla` | NEON | 8×4 | `fmla v.4s` | 32 v-reg, 128-bit |
| `SveFmla` | SVE | `VL/4 × 4` | `fmla z.s, p/z` | 可变长度, 32 z-reg |

**NeonFmla 微核**: MR=8 (2×NEON lane), NR=4。`ld2 {v0.4s, v1.4s}, [src]` 加载 2 行 × 4 列，`fmla` 累加。分块策略与 Avx2 相同 (BLIS 三级)。

**SveFmla 微核**: MR = SVE vector length / 4 bytes。例如 VL=256-bit → MR=8, VL=512-bit → MR=16。predicate register 实现尾部掩码（无 need for 尾部循环）。

### 1.3 NVIDIA GPU GEMM

| 变体 | SM 版本 | 关键指令 | 共享内存策略 | Tensor Core |
|------|--------|---------|-------------|-------------|
| `Sm70Wmma` | SM70 (V100) | `wmma.mma_sync.aligned` | 2 stage, 同步拷贝 | 16×16×16 F16 |
| `Sm80MmaAsync` | SM80 (A100) | `mma.sync.aligned` + `cp.async.ca.global` | 3+ stage, 异步拷贝 | 16×8×16 / 8×16×16 BF16 |
| `Sm90WgmmaTma` | SM90 (H100) | `wgmma.mma_async` + `cp.async.bulk.tensor` | TMA + cluster shared | 64×64×16 / 64×96×16 BF16 |
| `Sm100Tcgen05` | SM100+ (B100+) | `tcgen05` 系列 | TMEM + TMA | 下一代 tensor core |

**Sm70Wmma**: `wmma::load_matrix_sync` 加载 fragment → `wmma::mma_sync` 计算 → `wmma::store_matrix_sync` 写回。2 stage double buffering。

**Sm80MmaAsync**:
```
// 3-stage 异步流水线
cp.async.ca.global [smem_stage_0], [gmem_ptr], 16_bytes  // stage 0 拷贝
wait_group 2                                                // 等 stage 0 完成
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32       // stage 0 计算
cp.async.ca.global [smem_stage_1], [gmem_ptr], 16_bytes  // stage 1 拷贝
...
```

**Sm90WgmmaTma**:
```
// TMA 描述符 + wgmma 协作式
TmaDescriptor desc = create_tma_descriptor(A_tensor, tile_shape)
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem], [desc, coord], [mbarrier]
wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf32
    {%r0, ..., %r127}, %a_desc, %b_desc, %ctrl
// 多 wave: 同一 cluster 内多个 block 协作一个 GEMM tile
```

**多 wave 并发**: `num_waves` 控制 Thread Block Cluster 大小。`cluster_dims = (X, 1, 1)` 表示 X 个 block 在 X 维度协作。每个 block 负责 M 维度的 1/X 切片。

### 1.4 AMD GPU GEMM (RDNA 3/4)

| 变体 | 架构 | 关键指令 | WMMA 格式 | 适用条件 |
|------|------|---------|----------|---------|
| `Rdna3Wmma` | RDNA 3 (GFX11) | `v_wmma_f32_16x16x16_f16` | F16 × F16 → F32 | Wave32 |
| `Rdna4WmmaF16` | RDNA 4 (GFX12) | `v_wmma_f32_16x16x16_f16` | F16 × F16 → F32 | Wave32 |
| `Rdna4WmmaBf16` | RDNA 4 (GFX12) | `v_wmma_f32_16x16x16_bf16` | BF16 × BF16 → F32 | Wave32 |
| `Rdna4WmmaA4W8` | RDNA 4 (GFX12) | `v_wmma_f32_*_a4w8` | A4 × W8 → F32 | 硬件隐式 dequant |

**RDNA 4 a4w8 WMMA 详解**:
```
数学语义: C[f32] += dequant(A[a4], scale_a) × dequant(B[w8], scale_b)
硬件行为: 4-bit × 8-bit 矩阵乘，F32 累加
          scale 由硬件从附加元数据自动读取
          零软件 dequant — 权重保持原始压缩格式
内存效益: 4-bit 权重 = 0.5 bytes/elem vs F32 = 4 bytes/elem → 8× 内存节省
```

a4w8 不是独立指令，而是 WMMA 指令的操作数格式属性。code 发射时在 WMMA 指令的操作数位置指定 a4 和 w8 格式，硬件自动完成 dequant+multiply+accumulate。

## 2. Attention 微核指令矩阵

### 2.1 CPU Attention

| 变体 | ISA | 分块策略 | 关键指令 |
|------|-----|---------|---------|
| `ScalarTiled` | 基准 | Q tile × K tile | `mulss` + `addss` 循环 |
| `Avx2Tiled` | AVX2 | tile_q=64, tile_k=64 | `vfmadd231ps`, `vmulps`, softmax 向量化 |
| `Avx512Tiled` | AVX-512 | tile_q=128, tile_k=64 | zmm 全宽度 softmax (exp 向量化) |
| `PagedTiled` | 任意 | page-aware | 同上 + page offset 计算 |

**CPU tiled attention 核心循环**:
```
for qi in 0..seq_len step tile_q:
    load Q[qi..qi+tile_q, :] → registers/L1
    max_score = -inf
    sum_exp = 0
    for ki in 0..total_seq step tile_k:
        load K[ki..ki+tile_k, :] → L1
        S = Q × K^T  // 向量化点积
        max_score = max(max_score, max(S))
        sum_exp += sum(exp(S - max_score))
        accum += exp(S - max_score) × V[ki..ki+tile_k, :]
    output[qi] = accum / sum_exp
```

### 2.2 GPU Attention (Flash)

| 变体 | SM 版本 | 分块策略 | 共享内存使用 | 关键特性 |
|------|--------|---------|-------------|---------|
| `FlashV1` | SM80+ | block_q=64, block_k=64 | ~48 KB (Q+K+V+O) | online softmax, 同步拷贝 |
| `FlashV2` | SM90+ | block_q=128, block_k=128 | ~80 KB (TMA tile) | TMA 异步加载, swizzle, 更大 tile |

**Flash Attention online softmax**: 不需要完整 QK^T 矩阵，逐 K block 计算 running max 和 running sum，避免 O(N²) 内存。

## 3. Fused Dequant 指令矩阵

### 3.1 Q4/Q8 CPU Fused Dequant

**Q4_0 解包序列** (在 GEMM 微核 K-loop 内):
```
// 在 GEMM K 循环内部，每次读取 block_size 个 4-bit 权重
load 32 bytes → 64 个 4-bit 值    // 1 个 cache line
unpack_hi_4bit → 32 × int8       // 高 4 位
unpack_lo_4bit → 32 × int8       // 低 4 位
convert_int8_to_f32 → 64 × f32   // 符号扩展 + 转换
broadcast scale → f32            // block scale
mul_f32(unpacked, scale) → 64 × f32  // 反量化
// 紧接着执行正常的 GEMM 乘累加
vfmadd231ps accum, input, dequant_weight
```

**Q8_0 解包序列**:
```
load 32 bytes → 32 个 int8 值
convert_int8_to_f32 → 32 × f32
broadcast scale → f32
mul_f32(converted, scale) → 32 × f32
vfmadd231ps accum, input, dequant_weight
```

**内存布局**: Q4 权重按 block 存储，每个 block = `[block_size/2 bytes weights] + [4 bytes scale]`。GEMM K 循环步长 = block_size 而非 1。

### 3.2 MXFP4 双指针解包

MXFP4 格式: weights 和 scales 存储在相邻但独立的内存区域 (ARCH-MXFP4-SEPARATE):
```
blocks_ptr  → [block0, block1, ..., blockN]  // 每个 block = 32 × 4-bit = 16 bytes
scales_ptr  → [scale0, scale1, ..., scaleN]   // 每个 scale = 1 × f32 = 4 bytes

// GEMM K 循环内的解包:
load 16 bytes from blocks_ptr → 32 × 4-bit 值
unpack → 32 × int4 (0-15)
load 4 bytes from scales_ptr → 1 × f32 (shared scale)
// E2M1 MXFP4: value = int4 * scale * (2^(-k_bits))...
dequant_mxpf4(unpacked, scale) → 32 × f32
vfmadd231ps accum, input, dequant_mxpf4
```

### 3.3 硬件隐式 Dequant

| 硬件 | 指令 | 输入格式 | 累加格式 | 说明 |
|------|------|---------|---------|------|
| AMD GFX12 | `v_wmma_*_a4w8` | 4-bit × 8-bit | F32 | 硬件自动 dequant，零软件开销 |
| NVIDIA SM80+ | `mma.sync.*.s32.u8.u8.s32` | UINT8 × UINT8 | INT32 | IMMA 整数 tensor core |
| NVIDIA SM90+ | `mma.sync.*.f32.e4m3fxp.e4m3fxp.f32` | FP8 E4M3 × FP8 E4M3 | F32 | FP8 tensor core |

**GFX12 a4w8 详细流程**:
```
// 权重保持 Q4 × Q8 格式，不需要 dequant 步骤
// scale 元数据存储在权重旁，硬件自动读取
v_wmma_f32_16x16x16_a4w8
    C[0..15][0..15],    // F32 累加器
    A[0..15][0..15],    // 4-bit 输入 (保持原始格式)
    B[0..15][0..15]     // 8-bit 输入 (保持原始格式)
// 硬件内部完成: dequant(A[4bit]) × dequant(B[8bit]) + C → C
// 对软件透明 — 等同于 F32 GEMM 的数学结果
```

## 4. 向量化 Elementwise / Norm 指令矩阵

### 4.1 Elementwise (TraceOp → ISA)

| TraceOp | x86 AVX2 | x86 AVX-512 | AArch64 NEON | AArch64 SVE |
|---------|----------|-------------|-------------|-------------|
| Add | `vaddps ymm` | `vaddps zmm` | `fadd v.4s` | `fadd z.s p/m` |
| Mul | `vmulps ymm` | `vmulps zmm` | `fmul v.4s` | `fmul z.s p/m` |
| Fma | `vfmadd231ps ymm` | `vfmadd231ps zmm` | `fmla v.4s` | `fmla z.s p/m` |
| Exp | 多项式 (~12 条) | 多项式 (~12 条) | 多项式 (~14 条) | 多项式 (~14 条) |
| Sqrt | `vsqrtps ymm` | `vsqrtps zmm` | `fsqrt v.4s` | `fsqrt z.s p/m` |
| Max | `vmaxps ymm` | `vmaxps zmm` | `fmax v.4s` | `fmax z.s p/m` |
| Tanh | 有理逼近 (~20 条) | 有理逼近 (~20 条) | 有理逼近 (~20 条) | 有理逼近 (~20 条) |

### 4.2 Norm (RmsNorm / LayerNorm)

**RmsNorm 向量化序列**:
```
// Pass 1: sum of squares
vxorps acc, acc             // sum_squares = 0
for i in 0..hidden step simd_width:
    vload x → ymm           // load input[i..i+width]
    vfmadd231ps acc, x, x   // sum_squares += x²
vhaddps acc → scalar        // horizontal reduce
div ss, hidden               // mean = sum / hidden
add ss, eps                  // mean + eps
sqrtps ss → inv_rms         // sqrt
div const_1, inv_rms → inv_rms  // 1/rms

// Pass 2: scale and apply weight
broadcast inv_rms → ymm
for i in 0..hidden step simd_width:
    vload x → ymm
    vload w → ymm_weight    // norm weight
    vmulps x, inv_rms → ymm // x * inv_rms
    vmulps ymm, ymm_weight → ymm  // * weight
    vstore ymm → output[i]
```

**LayerNorm 差异**: Pass 1 先算 mean (sum/hidden)，再算 variance (sum((x-mean)²)/hidden)，Pass 2 用 `(x-mean)/sqrt(var+eps)*weight+bias`。

## 5. 设备能力探测方法

### 5.1 x86_64 探测

| 能力 | 探测方法 | 对应 DeviceProfile 字段 |
|------|---------|----------------------|
| AVX2 | CPUID.07H:EBX[5] | `has_avx2` |
| AVX-512 F | CPUID.07H:EBX[16] | `has_avx512` |
| AVX-512 BW | CPUID.07H:EBX[30] | `has_avx512_bw` |
| AVX-512 VNNI | CPUID.07H:ECX[11] | `has_avx512_vnni` |
| AVX-512 BF16 | CPUID.07H_1:EAX[5] | `has_avx512_bf16` |
| AMX | CPUID.07H:EDX[24] | `has_amx` |
| Cache sizes | CPUID.04H / CPUID.8000001DH | `l1_cache_bytes`, `l2_cache_bytes`, `l3_cache_bytes` |
| Core count | `/proc/cpuinfo` or `sched_getaffinity` | `num_cores` |

### 5.2 AArch64 探测

| 能力 | 探测方法 | 对应 DeviceProfile 字段 |
|------|---------|----------------------|
| NEON | `hwcap` (always present on ARMv8+) | `has_neon` |
| SVE | `hwcap` HWCAP_SVE | `has_sve` |
| SVE VL | `prctl(PR_SVE_GET_VL)` | `sve_vl_bytes` |
| DotProd | `hwcap` HWCAP_ASIMDDP | `has_dotprod` |
| I8MM | `hwcap` HWCAP_I8MM | `has_i8mm` |
| BF16 | `hwcap` HWCAP_BF16 | `has_bf16` |

### 5.3 NVIDIA GPU 探测

| 能力 | 探测方法 | 对应 DeviceProfile 字段 |
|------|---------|----------------------|
| SM 版本 | `cudaGetDeviceProperties().major*10+minor` | `tensor_core_gen` |
| SM 数量 | `cudaGetDeviceProperties().multiProcessorCount` | `num_sm` |
| 共享内存 | `cudaGetDeviceProperties().sharedMemPerBlock` | `shared_mem_bytes` |
| 寄存器/SM | `cudaGetDeviceProperties().regsPerBlock` | `regs_per_sm` |
| 内存带宽 | `cudaGetDeviceProperties().memoryBusWidth` | `mem_bus_width` |

### 5.4 AMD GPU 探测

| 能力 | 探测方法 | 对应 DeviceProfile 字段 |
|------|---------|----------------------|
| GFX 架构 | `hipGetDeviceProperties().gcnArchName` | `gpu_arch: Gfx11/Gfx12` |
| WMMA 支持 | 检查 gfx arch >= gfx1100 (RDNA 3) | `has_wmma` |
| a4w8 格式 | 检查 gfx arch >= gfx1200 (RDNA 4) | `has_wmma_a4w8` |
| CU 数量 | `hipGetDeviceProperties().multiProcessorCount` | `num_cu` |
| LDS 大小 | `hipGetDeviceProperties().sharedMemPerBlock` | `lds_bytes` |

## 6. 交叉引用

| 主题 | 位置 |
|------|------|
| DeviceProfile 字段定义 | `SPEC/02-HARDWARE.md` §2 |
| ComputeStrategy 枚举 (GemmVariant/...) | `SPEC/01-JIT-PIPELINE.md` §5 |
| 策略选择算法 | `SPEC/01-JIT-PIPELINE.md` §5.11 |
| ISA Lowering 流程 | `SPEC/01-JIT-PIPELINE.md` §6 |
| 融合策略差异矩阵 | `SPEC/04-OPERATORS.md` §8-10 |
| NO_HW_DEGRADATION 铁律 | `SPEC/00-PHILOSOPHY.md` |
