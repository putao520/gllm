# 算子与算子分发

> **SSOT**: 本文档是 gllm + gllm-kernels 统一项目算子库、算子语义定义、三层分发架构、硬件特化路径、融合策略差异矩阵、统一 GEMM 模型、硬件参数→codegen 贯通链的唯一真源。
>
> 交叉引用: `01-JIT-PIPELINE.md`（JIT 管线）、`02-HARDWARE.md`（DeviceProfile）、`00-PHILOSOPHY.md`（铁律）

## 1. 三层零成本分发架构

算子通过三层分发实现零运行时开销:

| 层次 | 分发机制 | 开销 | 实现位置 |
|------|---------|------|---------|
| **Layer 1: Backend** | 编译时 feature gate 单态化 | 零 | `#[cfg(feature = "jit-cuda")]` |
| **Layer 2: ISA** | 启动时 `DeviceProfile::detect()` 一次性探测 | 零（编译时常量） | `gllm-kernels/src/compiler/` |
| **Layer 3: Precision** | 编译时 `<E: Element>` 泛型单态化 | 零 | Backend trait impl |

```rust
错误: match dtype { F16 => path_a(), F32 => path_b() }
正确: <E: Element> fn compute(input: &[E]) -> E
```

### 1.1 Backend 分发

| 后端 | 类型 | Feature Gate | 实现位置 |
|------|------|-------------|---------|
| **CpuBackend** | CPU (JIT 编译内核) | 默认 | `src/compat/cpu_backend.rs` |
| **CudaBackend** | NVIDIA GPU | `jit-cuda` | `src/compat/cuda_backend.rs` |
| **HipBackend** | AMD GPU (ROCm) | `jit-hip` | `src/compat/hip_backend.rs` |
| **MetalBackend** | Apple GPU | `jit-metal` | `src/compat/metal_backend.rs` |

后端检测优先级: CUDA -> ROCm -> Metal -> CPU (`src/backend/detection.rs`)

### 1.2 ISA 分发 (Layer 2)

启动时通过 CPUID / /proc 检测 ISA，结果缓存于 `DeviceProfile`（完整字段定义见 `SPEC/02-HARDWARE.md` §2）。运行时通过 `DeviceProfile` 字段选择代码路径。

| ISA | 寄存器宽度 | 典型平台 |
|-----|---------|---------|
| **AVX2** | 256-bit, 16 ymm | Haswell+ (2013) |
| **AVX-512** | 512-bit, 32 zmm | Skylake-X+ (2017) |
| **NEON** | 128-bit, 32 v-reg | ARMv8+ 全系 |
| **SVE** | 128~2048-bit, 32 z-reg | Neoverse V1+ |
| **AMX** | Tile | Intel Xeon Scalable |

硬件参数通过 `DeviceProfile` 字段驱动 codegen: `simd_width`、`use_avx512`、`has_amx`、`has_sve`、`sve_vl_bytes`、`cache_sizes` 等。

### 1.3 Precision 分发 (Layer 3)

精度通过 `<E: Element>` 编译时单态化。运行时无 `match dtype` 分支。`Element` trait 定义算术运算接口，精度差异体现在类型大小和指令选择上，由 codegen 处理。

| DType | Element 实现 | 字节 | SIMD 指令集 |
|------|---------------|------|---------|
| **F32** | `f32` | 4 | AVX2 `vaddps` / AVX-512 `vaddps` / NEON `fadd` |
| **F16** | `half::f16` | 2 | AVX2 F16C `vcvtph2ps` / AVX-512 FP16 `vaddph` / NEON FP16 `fadd` |
| **BF16** | `half::bf16` | 2 | AVX-512 BF16 `vdpbf16ps` / NEON BF16 `bfdot` |

## 2. Element trait

Element trait 定义标量算术和类型约束，供 codegen 使用:

```rust
pub trait Element: Copy + Clone + Default + Sized + Send + Sync + Debug {
    const ZERO: Self;
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
    fn pack_i8(v: Self) -> [u8];
    fn unpack_i8(data: &[u8]) -> Self;
}
```

实现: `f32`, `half::f16`, `half::bf16`。编译时单态化确保零运行时分支。

## 3. Backend trait

Backend trait 定义硬件后端的统一接口。每个方法通过 `<E: Element>` 单态化。

| 方法 | 签名概要 |
|------|---------|
| `rms_norm` | `(x: &[E], out: &mut [E], weight: &[E], eps: f32)` |
| `rope` | `(q: &mut [E], k: &mut [E], pos: usize, head_dim: usize, ...)` |
| `gemm` | `(a: &[E], b: &[E], c: &mut [E], m: usize, n: usize, k: usize)` |
| `gemv` | `(a: &[E], x: &[E], y: &mut [E], m: usize, n: usize)` |
| `softmax` | `(x: &mut [E], len: usize)` |
| `silu` | `(x: &mut [E], len: usize)` |
| `swiglu` | `(gate: &[E], up: &[E], out: &mut [E], len: usize)` |
| `embed` | `(input: &[u8], output: &mut [E], ...)` |
| `dequantize_row` | `(data: &[u8], scale: f32, zero_point: i32, ...)` |

## 4. 算子清单

### 4.1 BLAS 模块

| 算子 | 数学定义 |
|------|------|
| `gemm` | `C = A * B + bias` |
| `gemv` | `y = A * x` |
| `vec_dot` | 内积 |
| `vec_add` | 向量加 |
| `vec_mul` | 向量乘 |
| `vec_scale` | 标量乘 |
| `vec_axpy` | AXPY: y += a * x |

BLIS GEMM 五层级循环: Pack A/B -> K-loop (MC) -> IR 循环 (NC) -> Tile (MR/NR) -> 输出。

### 4.2 激活函数

| 算子 | 数学定义 |
|------|------|
| `silu` | `x * sigmoid(x)` |
| `gelu` | `x * Phi(x)` |
| `relu` | `max(0, x)` |
| `tanh` | `tanh(x)` |
| `softmax` | `exp(x_i) / sum(exp(x_j))` |
| `swiglu` | `silu(gate) * up` |
| `exp` | `e^x` |

### 4.3 归一化

| 算子 | 数学定义 |
|------|------|
| `rms_norm` | `x / sqrt(mean(x^2) + eps) * weight` |
| `layer_norm` | `(x - mean) / std * weight + bias` |

### 4.4 位置编码

| 算子 | 数学定义 |
|------|------|
| `rope` | 旋转位置编码 |
| `rope_with_pos` | 带显式位置的 RoPE |

### 4.5 注意力

| 算子 | 说明 |
|------|------|
| `flash_attention` | 融合注意力: Softmax + 因果掩码 + 输出投影 + KV cache |
| `gqa` | 分组查询注意力: KV cache 读取 + GQA 分组 + 缩放 |
| `paged_attention` | 分页注意力: 虚拟内存管理 |
| `sliding_window` | 滑动窗口注意力 |

### 4.6 MoE (Mixture of Experts)

| 算子 | 数学定义 | 输入 | 输出 |
|------|---------|------|------|
| `MoEGate` | `router_logits = hidden @ weight.T; probs = softmax(router_logits); top_k_indices, top_k_weights = topk(probs)` | hidden [seq, hidden_size], weight [hidden_size, num_experts] | router_weights [seq, top_k], router_indices [seq, top_k] |
| `MoERouter` | `logits = hidden @ weight.T + bias; probs = softmax(logits); weights, indices = top_k(probs, k)` | hidden [seq, hidden_size], weight [hidden_size, num_experts], bias [num_experts] | router_weights [seq, top_k], router_indices [seq, top_k] |
| `MoEDispatchPacked` | `for each token: selected_experts = indices[token]; for each expert: out += expert_ffn(hidden, expert_weights) * router_weight` | hidden_input, router_weights, router_indices, gate_up_blocks, gate_up_scales, gate_up_bias, down_blocks, down_scales, down_bias (9 个) | output [seq, hidden_size] |

**MoERouter vs MoEGate**:
- `MoEGate`: DeepSeek-V3 风格，无 bias，softmax 后 top-k。用于 `OpKind::MoEGate`
- `MoERouter`: GPT-OSS-20B 风格，有 bias，softmax 后 top-k。用于 `OpKind::MoERouter`
- 两者输出相同格式：(router_weights, router_indices)，供下游 `MoEDispatchPacked` 消费

**MoERouter JIT lowering 策略**:
```
MoERouter 分解为 3 个子操作:
  1. GEMM: hidden @ weight.T → router_logits [seq, num_experts]
  2. Softmax: softmax(router_logits) → router_probs [seq, num_experts]
  3. TopK: top_k(router_probs, k) → (weights, indices) [seq, top_k]

JIT 实现: lower_moe_router() 内部依次调用:
  - emit_gemm_inline_with_hook() (GEMM)
  - lower_reduction_softmax() (Softmax)
  - lower_topk() (TopK — 新增 VmInstr::TopK)
```

**ComputePattern 分类**: `OpSemantics::Gemm`（内含 GEMM 子操作）
**lower 函数**: `lower::lower_moe_router()` — 专用 lower，内部组合 GEMM + softmax + top-k

## 5. 量化类型支持

### 5.1 GgmlDType 到 QuantType 映射

GGUF 格式 (21 种) 通过 `GgmlDType` 枚举标识，映射到 `QuantType` 后分发到对应的解量化内核。

| GgmlDType | QuantType | 分发目标 |
|----------|-----------|---------|
| `Q4_0` | `Q4_0` | `classic_matmul` |
| `Q4_1` | `Q4_1` | `classic_matmul` |
| `Q5_0` | `Q5_0` | `classic_matmul` |
| `Q5_1` | `Q5_1` | `classic_matmul` |
| `Q8_0` | `Q8_0` | `classic_matmul` |
| `Q8_1` | `Q8_1` | `classic_matmul` |
| `Q2_K` | `Q2K` | `kquant_matmul` |
| `Q3_K` | `Q3K` | `kquant_matmul` |
| `Q4_K` | `Q4K` | `kquant_matmul` |
| `Q5_K` | `Q5K` | `kquant_matmul` |
| `Q6_K` | `Q6K` | `kquant_matmul` |
| `Q8_K` | `Q8K` | `kquant_matmul` |
| `IQ1_S` | `IQ1S` | `iq_matmul` |
| `IQ1_M` | `IQ1M` | `iq_matmul` |
| `IQ2_XXS` | `IQ2XXS` | `iq_matmul` |
| `IQ2_XS` | `IQ2XS` | `iq_matmul` |
| `IQ2_S` | `IQ2S` | `iq_matmul` |
| `IQ3_XXS` | `IQ3XXS` | `iq_matmul` |
| `IQ3_S` | `IQ3S` | `iq_matmul` |
| `IQ4_NL` | `IQ4NL` | `iq_matmul` |
| `IQ4_XS` | `IQ4XS` | `iq_matmul` |
| `AWQ4` | `AWQ4` | `awq_matmul` |
| `GPTQ4` | `GPTQ4` | `gptq_matmul` |

### 5.2 解量化内核

| 内核 | 格式类别 | 说明 |
|------|---------|------|
| `classic_matmul` | Classic (Q4_0/Q5_0/Q8_0) | block_size = 32, 最简单的量化格式 |
| `kquant_matmul` | K-Quant (Q2_K~Q8_K) | block_size = 256, 高精度量化 |
| `iq_matmul` | IQ 系列 (IQ1~IQ4) | block_size = 256, 不量化矩阵 |
| `awq_matmul` | AWQ4 | block_size = 128, 激活感知量化 |
| `gptq_matmul` | GPTQ4 | block_size = 128, 静态后训练量化 |

### 5.3 TurboQuant 静态极化

TurboQuant 将权重量化到 W4A4 或 W8A8 模式，JIT 编译时锁定执行路径，推理过程中无类型判断分支。

| 模式 | 权重精度 | 量化方式 | 适用格式 |
|------|---------|---------|---------|
| W4A4 | 4-bit (非对称) | 非对称量化 + FWHT 在线旋转 | GGUF Q4_K/Q5_K |
| W8A8 | 8-bit (非对称) | 非对称量化 + RaBitQ 无偏修正 | GGUF Q8_K |

TurboQuant 严格受限于 GGUF 格式权重。SafeTensors/ONNX 原生浮点权重直接使用对应的 DType 模式。禁止运行时动态切换 TurboQuant 模式。运行时数学优化（FWHT 在线旋转、KIVI 非对称量化、RaBitQ 无偏修正、Dual-Track 显存池）详见 `SPEC/05-OPTIMIZATIONS.md` §4。

## 6. GPU SM 版本分发

### 6.1 PtxKernelRegistry

GPU 后端通过 `PtxKernelRegistry` 按 SM 版本选择最优 PTX 内核。`SmRange` 定义 SM 版本范围。

```rust
pub struct SmRange {
    pub min_sm: u32,
    pub max_sm: u32,
}
```

查找策略: 目标 SM 版本在注册表项的范围内时返回该项，否则返回错误。禁止 SM 版本降级。

### 6.2 SM 版本特化

| SM 版本 | 矩阵乘指令 | 异步预取 | 典型平台 |
|--------|---------|---------|---------|
| sm_100+ | `tcgen05.mma` (block-scaled) | TMA 2D/5D | Blackwell |
| sm_90 | `WGMMA` | TMA 2D/5D prefetch | Hopper |
| sm_80~89 | `mma.sync` | `cp.async` | Ampere/Ada |
| sm_70~79 | `wmma` | 无 | Volta/Turing |

### 6.3 HIP AMD GPU

| gfx_arch | GPU 架构 | tensor_core_gen | 典型平台 |
|---------|---------|-----------------|---------|
| gfx940+ | CDNA3 | 3 (WMMA) | MI300 |
| gfx908+ | CDNA2 | 2 (WMMA) | MI250 |
| gfx1100+ | RDNA3 | 1 | RX 7900 |

warp_size: RDNA (gfx1000+) = 32, CDNA = 64。

### 6.4 Metal Apple GPU

Apple GPU 通过 Metal framework 直接编译。所有 Apple GPU 共享 `MetalBackend` 实现。

## 7. 硬件特化路径

### 7.1 x86_64 特化

| 组件 | 特化机制 |
|------|---------|
| BLIS GEMM | 5 层级循环: Pack A/B -> K-loop (MC) -> IR 循环 (NC) -> Tile (MR/NR) -> 输出 |
| 手写汇编微内核 | AVX2: 6×16, AVX-512: 14×32 (`global_asm!`) |
| AVX-512 BF16 | `vdpbf16ps` 原生 BF16 内积 |
| AVX-512 VNNI | `vpdpbusd` W8A8 INT8 内积 |
| AMX tile | Intel AMX `tdpbssd` 定点规格 |

### 7.2 ARM 特化

| 组件 | 特化机制 |
|------|---------|
| BLIS GEMM | 5 层级循环: Pack A/B -> K-loop -> Tile -> 输出 |
| 手写汇编微内核 | NEON: 8×12 (`global_asm!`) |
| SVE 向量 | 可变长度 128~2048-bit |
| NEON BF16 | `bfdot` BF16 内积 |

### 7.3 GPU 特化

| 组件 | SM 版本 | 特化机制 |
|------|---------|---------|
| TMA prefetch | sm_90+ | 张量内存加速器异步预取 |
| WGMMA | sm_90+ | Warp 级矩阵乘累加 |
| tcgen05.mma | sm_100+ | Block-scaled 矩阵乘累加 |
| FP4/FP6 native | sm_100+ | 原生低精度 MMA |

## 8. 融合策略硬件差异矩阵

Phase 2 融合决策由 `DeviceProfile` 驱动，不同硬件平台的融合能力差异显著。本节定义每种融合模式在各硬件 Profile 上的行为和约束。

### 8.1 EpilogueInjection 融合深度

GEMM 微内核累加完毕后，Epilogue 在累加器寄存器上原地执行 TraceOp 链。融合深度受可用 scratch 寄存器数限制:

| 硬件 Profile | 微内核 MR×NR | 累加器寄存器 | Scratch 寄存器 | 最大 Epilogue 深度 | 说明 |
|-------------|------------|------------|---------------|-------------------|------|
| AVX2 | 6×16 | 12 ymm | 4 ymm | 2-3 ops | 16 ymm 总量，寄存器溢出到栈后性能骤降 |
| AVX-512 | 14×32 | 28 zmm | 4 zmm | 6-8 ops | 32 zmm 总量，充裕 scratch 空间 |
| AMX (AVX-512) | 14×32 | 28 zmm | 4 zmm | 8+ ops | AMX tile 做主 GEMM，zmm 全用于 epilogue |
| NEON | 8×12 | 24 v | 8 v | 4-6 ops | 32 v-reg 总量，适中 scratch |
| SVE2 | 动态 | 动态 | 动态 | 4-8 ops | 按 `sve_vl_bytes` 动态计算 |
| GPU sm_70 | wmma 16×16×16 | — | — | 3-4 ops | 寄存器压力限制融合深度 |
| GPU sm_80 | mma.sync 16×8×16 | — | — | 4-6 ops | 较多寄存器支持更深 epilogue |
| GPU sm_90 | WGMMA 16×16×64 | — | — | 6-8 ops | Warp Specialization 释放寄存器压力 |
| GPU sm_100+ | tcgen05.mma | — | — | 8+ ops | TMEM 替代 shared memory 做累加 |

### 8.2 TileLevelFusion vs ComputeRoot 决策

由 DeviceProfile 的 cache 层级数据驱动:

```
if predecessor_output_bytes > profile.l1_cache_bytes * 0.75 {
    TileLevelFusion { tile_rows: mc }  // 嵌入 GEMM MC 循环
} else {
    ComputeRoot  // 先算完驻留 L1
}
```

| 硬件 Profile | L1 Cache (KB) | TileLevelFusion 阈值 | ComputeRoot 范围 |
|-------------|-------------|---------------------|-----------------|
| AVX2 | 32 | > 24 KB | hidden ≤ 6K elem (F32) |
| AVX-512 | 32 | > 24 KB | hidden ≤ 6K elem (F32) |
| NEON | 32-64 | > 24-48 KB | hidden ≤ 6-12K elem (F32) |
| GPU sm_80 | 48-164 (SMEM) | > SMEM × 75% | tile ≤ 36-123 KB |
| GPU sm_90 | 228 (group SMEM) | > 171 KB | tile ≤ 171 KB |

### 8.3 QkvSharedInput 适用性

Q/K/V 三个 GEMM 共享 pack_a，消除重复 pack 开销:

| 硬件 Profile | pack_a 开销 | 融合收益 | 说明 |
|-------------|-----------|---------|------|
| AVX2 | 高 (6×16 tile pack) | 显著 | 3 次 pack → 1 次 |
| AVX-512 | 高 (14×32 tile pack) | 显著 | 3 次 pack → 1 次 |
| AMX | 低 (tile load) | 中等 | AMX tile load 天然高效 |
| GPU sm_80 | 中 (cp.async) | 显著 | 3 次 Global→Shared → 1 次 |
| GPU sm_90 | 低 (TMA 2D) | 中等 | TMA prefetch 天然高效 |

### 8.4 NormIntoGemm 融合

RmsNorm 输出直接喂入 GEMM（无中间写回），与 TileLevelFusion 协同:

| 硬件 Profile | Norm 计算位置 | 数据流 |
|-------------|-------------|-------|
| CPU 全系 | 寄存器内 Norm → pack_a 直通 | Norm 结果写入 scratchpad 的 normed 区域，紧接着被 pack_a 消费 |
| GPU sm_80 | Block 内 Norm → SMEM 直通 | Thread Block 内完成 Norm，结果写入 SMEM 供 mma.sync 消费 |
| GPU sm_90 | Warp 内 Norm → Register 直通 | Warp Specialization producer 完成 Norm，consumer WGMMA 直接消费 |
| GPU sm_100+ | TMEM 内 Norm → TMEM 直通 | TMEM 完成累加和 Norm，无 SMEM 中间写回 |

### 8.5 FFNBlock 融合

Gate GEMM + Up GEMM 的 Gate/Up 结果融合激活 (SiLU) + 乘法:

| 硬件 Profile | 融合方式 | 说明 |
|-------------|---------|------|
| AVX2 | 分离 GEMM + LoopFusion(SiLU×Up) | Gate/Up 各自独立 GEMM，epilogue 仅处理 SiLU |
| AVX-512 | EpilogueInjection(SiLU) + LoopFusion(Mul) | Gate GEMM epilogue 内联 SiLU，结果与 Up 做 LoopFusion 乘法 |
| GPU sm_80 | EpilogueInjection(SiLU) + Elementwise | Gate epilogue 内联 SiLU，与 Up 做 fused multiply |
| GPU sm_90 | EpilogueInjection(SiLU+Mul) | Warp Specialization 支持 Gate epilogue 内同时完成 SiLU 和乘法 |

### 8.6 CrossLayerResidual 融合

Add → RmsNorm scratchpad 直通:

| 硬件 Profile | 融合方式 | 说明 |
|-------------|---------|------|
| CPU 全系 | Scratchpad 直通 | Add 结果写入 RmsNorm scratchpad，无中间写回 |
| GPU sm_80 | SMEM 直通 | Thread Block 内 Add→Norm 无 Global 写回 |
| GPU sm_90 | Register 直通 | Warp Specialization consumer 直接消费 Add 结果 |

## 9. 算子实现硬件差异矩阵

### 9.1 GEMM 微内核规格

| ISA | MR×NR | 累加器寄存器 | Scratch 寄存器 | 数据搬运 |
|-----|-------|------------|---------------|---------|
| AVX2 | 6×16 | 12 ymm | 4 ymm | pack_a/pack_b (F32) |
| AVX-512 | 14×32 | 28 zmm | 4 zmm | pack_a/pack_b (F32) |
| AVX-512 BF16 原生 | 14×32 | 28 zmm | 4 zmm | pack_b K-pair interleaved, pack_a BF16 列主序 |
| AMX tile | 16×16 (tile) | 8 TMM | — | tile load/store (TDPBF16PS) |
| NEON | 8×12 | 24 v | 8 v | pack_a/pack_b (F32) |
| GPU sm_70 wmma | 16×16×16 | — | — | global_load (无异步) |
| GPU sm_80 mma.sync | 16×8×16 | — | — | cp.async 128B |
| GPU sm_90 WGMMA | 16×16×64 | — | — | TMA 2D prefetch |
| GPU sm_100+ tcgen05 | 16×16×64+ | — | — | TMA 2D/5D + TMEM |

### 9.2 Attention 实现路径

FlashAttention 按 SM 版本四路特化:

| SM 范围 | 名称 | 核心指令 | 数据搬运 | Softmax |
|---------|------|---------|---------|---------|
| sm_100+ | FA4 Block-Scaled | tcgen05.mma + block_scale + TMEM | TMEMLoad/TMEMScatter | Online softmax + block-scale factor |
| sm_90 | FA3 Pipeline | WGMMA 16×16×64 + TMA | TMA 2D prefetch + Warp Specialization | Online softmax in registers |
| sm_80-89 | FA2 Tiled | mma.sync 16×8×16 | cp.async 128B | Online softmax in shared memory |
| sm_70-79 | wmma Tiled | wmma 16×16×16 | global_load (无异步) | Online softmax in shared memory |

CPU Attention 路径:

| CPU Profile | Attention 实现 | 说明 |
|------------|---------------|------|
| AVX2 | Loop-based + vaddps/vmulps | 逐 head 计算，SIMD 向量化 Softmax |
| AVX-512 | Loop-based + zmm 向量 | 更宽的 SIMD Softmax，减少循环次数 |
| AMX | AMX tile QK^T + V 投影 | Tile 矩阵乘法做 Attention 核心 |
| NEON | Loop-based + NEON v | 128-bit 向量化 |
| SVE2 | Predicate-gated 向量化 | 可变长向量，无 padding 浪费 |

### 9.2a1 All-in-GEMM + Softmax 融合

Softmax 融合到 Attention 的 QK^T → V 投影 GEMM epilogue 中:

- **online softmax** 在累加器/共享内存中完成 max + exp + sum + scale
- **softmax_weights** 直接作为 V 投影的乘数因子
- 消除 `attn_output → softmax → O_proj` 三次独立内存读写
- 融合后: QK^T → [online softmax in regs/smem] → scale → × V → O_proj，单次 GEMM 完成

| 硬件 Profile | Softmax 融合方式 | 消除的中间结果 |
|------------|----------------|---------|
| AVX2 | ymm 寄存器内 online softmax → scale 嵌入 vmovups 输入 | attn_output + softmax + scale 三次内存读写 |
| AVX-512 | zmm 寄存器内 online softmax → scale 嵌入 vfmadd 输入 | 同上 |
| GPU sm_80 | mma.sync epilogue 内 softmax → scale 作为 V 投影乘数 | attn_output + softmax + O_proj 三次 GEMM launch |
| GPU sm_90 | WGMMA epilogue + Warp Specialization | softmax 在 consumer warp 完成 | 同上 |

### 9.3 RmsNorm 实现路径

### 9.3 RmsNorm 实现路径

| 硬件 Profile | 实现方式 | 指令 |
|-------------|---------|------|
| AVX2 | 两遍扫描 + ymm 向量 | vmulps + vaddps (reduce) + vdivps |
| AVX-512 | 两遍扫描 + zmm 向量 | vfmadd231ps (平方) + vreduce (求和) |
| NEON | 两遍扫描 + v 向量 | fmla + fadd (reduce) |
| SVE2 | Predicate-gated 单遍 | whilelt + fmla (无尾处理) |
| GPU 全系 | Warp reduce + scale | warp shuffle reduce + FMA scale |

### 9.4 RoPE 实现路径

| 硬件 Profile | 实现方式 | 说明 |
|-------------|---------|------|
| AVX2 | 查表 + ymm FMA | 预计算 cos/sin 表，vmulps + vfmadd231ps |
| AVX-512 | 查表 + zmm FMA | 更宽的 SIMD，一次处理更多 head_dim |
| NEON | 查表 + v FMA | fmul + fmla |
| GPU 全系 | Register 内计算 | SFU ex2.approx + fma.rn |

### 9.5 Softmax 实现路径

| 硬件 Profile | 实现方式 | 说明 |
|-------------|---------|------|
| AVX2 | 三遍 (max → exp → sum+scale) | 多项式逼近 exp，ymm reduce |
| AVX-512 | 三遍 + zmm reduce | 更宽的 reduce，减少迭代 |
| NEON | 三遍 + v reduce | fmaxv + exp 逼近 |
| GPU sm_80+ | Online softmax + mma.sync | 单遍 max+exp+sum 在寄存器 |
| GPU sm_90+ | Online softmax + WGMMA | Warp Specialization 内联 softmax |

### 9.6 量化 GEMV/GEMM 路径

| ISV 特性 | 指令 | 适用场景 |
|---------|------|---------|
| AVX-512 VNNI | `vpdpbusd` | INT8 量化 GEMV (W8A8) |
| AVX-512 BF16 | `vdpbf16ps` | BF16 原生 GEMM (pack_b K-pair interleaved) |
| AMX tile | `tdpbssd` / `tdpbf16ps` | INT8 / BF16 tile 矩阵乘 |
| NEON BF16 | `bfdot` | BF16 点积 |
| GPU Tensor Core | wmma/mma.sync/WGMMA | 全精度范围量化矩阵乘 |

CPU ISV 策略链 (`select_gemm_strategy()`):

```
AMX → oneDNN → Accelerate → JIT BLIS
```

## 10. 统一分层 GEMM 模型

所有平台的 GEMM 共享同一个分层 tiling 模型，每层映射到不同硬件资源:

| 层次 | CPU (AVX2) | CPU (AVX-512) | CPU (AMX) | GPU (CUDA sm_80) |
|------|-----------|--------------|-----------|-----------------|
| 最外层 | NC×KC (L3) | NC×KC (L3) | NC×KC (L3) | Grid tile |
| 中间层 | MC×KC (L2) | MC×KC (L2) | MC×KC (L2) | Block tile (shared mem) |
| 微内核 | 6×16 FMA | 14×32 FMA | 16×16 TDPBF16PS | Warp mma.sync |
| 数据搬运 | pack_a/pack_b | pack_a/pack_b | tile load/store | Global→Shared (cp.async) |

Phase 2 融合决策（EpilogueInjection、TileLevelFusion 等）对所有平台通用。差异只在 Phase 3 微内核指令选择。

### 10.1 GEMM 分块参数推导

DeviceProfile 的缓存层级直接驱动 GEMM 分块:

```
DeviceProfile.cache_sizes (L1, L2, L3)
  → GemmBlocking { kc, mc, nc }
    → Phase 3: NC/MC/KR 三重循环边界
```

- `kc` 适配 L1 Cache（K 维度分块）
- `mc` 适配 L2 Cache（M 维度分块）
- `nc` 适配 L3 Cache（N 维度分块）
- `mr`, `nr` 由微内核规格决定

### 10.2 GEMM 代码结构

Phase 3 生成的 GEMM + Epilogue 代码结构（CPU 和 GPU 共享同一 tiling 逻辑）:

```
prologue (save callee-saved)
├── NC loop
│   ├── pack_b
│   ├── MC loop
│   │   ├── [可选] tiled_predecessor: 嵌入的前驱算子 tile 计算
│   │   ├── pack_a
│   │   └── NR loop
│   │       └── 微内核:
│   │           ├── 累加器清零
│   │           ├── K-loop: FMA 序列
│   │           ├── [可选] epilogue: TraceOp → SIMD 指令
│   │           └── store
│   └── edge tile
epilogue (restore + ret)
```

## 11. 硬件参数→codegen 贯通链

DeviceProfile 的每个字段如何驱动 Phase 3 代码生成。

### 11.1 GEMM 分块参数

```
DeviceProfile.cache_sizes (L1, L2, L3)
  → GemmBlocking { kc, mc, nc }
    → Phase 3: NC/MC/KR 三重循环边界
```

### 11.2 寄存器分配

```
DeviceProfile.num_simd_regs
  → 微内核累加器数量 (MR × NR_vecs)
  → epilogue 可用 scratch 寄存器
  → FusionRule 是否允许 EpilogueInjection
```

### 11.3 预取距离

```
DeviceProfile (平台相关)
  → prefetch_distance: AVX2=256B, AVX-512=512B, NEON=128B
  → Phase 3: K-loop 中的 PREFETCHT0 指令偏移
```

### 11.4 SIMD 指令宽度

```
DeviceProfile.simd_width + use_avx512
  → Phase 3: ymm (256-bit) vs zmm (512-bit) 指令选择
  → 循环步长: step = simd_width / sizeof(elem)
```

### 11.5 GPU Tensor Core 代数

```
DeviceProfile.tensor_core_gen
  → GemmStrategy: JitGpuTensorCore (gen≥2) / JitGpu (gen<2)
  → Phase 3 PTX: wmma (sm70) / mma.sync (sm80) / WGMMA (sm90) / tcgen05.mma (sm100+)
```

### 11.6 ISV 特性→代码路径

```
DeviceProfile.has_bf16 + use_avx512 + gemm_dtype==BF16
  → use_native_bf16() = true
  → Phase 3: VDPBF16PS 原生 BF16 GEMM 路径 (pack_b/pack_a/fma_body _bf16_native 后缀)

DeviceProfile.has_vnni
  → Phase 3: INT8 量化 GEMV 使用 vpdpbusd

DeviceProfile.has_amx
  → Phase 3: GEMM 微内核使用 TDPBF16PS tile 指令
```

### 11.7 GPU 共享内存

```
DeviceProfile.shared_mem_per_block
  → Phase 2: TileLevelFusion 的 tile 大小上限
  → Phase 3 PTX: .shared .f32 声明大小
```

### 11.8 GPU SM 版本

```
DeviceProfile.sm_version
  → PtxKernelRegistry: 选择最优 emitter
  → PTX 版本: .version 7.0 (sm70), .version 8.0 (sm80), ...
  → 数据搬运: TMA (sm90+) / cp.async (sm80+) / global_load (sm70)
  → 同步: bar.sync / cuda::barrier
```

## 13. GEMM 极致优化基础

### 13.1 K-loop 软件流水线

GEMM K-loop 内 prefetch 与 compute 重叠，隐藏内存延迟：

```
K-loop iteration i:
  PREFETCHT0 A[k+pf_dist]     ← DRAM → L1, 异步（硬件预取不占执行槽）
  PREFETCHT0 B[k+pf_dist]     ← DRAM → L1, 异步
  vfmadd231ps acc, A[k], B[k]  ← 计算（数据已在 L1 命中）
```

**软件流水线深度**（同时飞行的 prefetch 数量）：

| 硬件 | K-loop depth | 预取距离 | 原因 |
|------|-------------|---------|------|
| AVX2 | 1 | 256B (4 cache lines) | 16 ymm 溢出压力，只能 prefetch 一行 |
| AVX-512 | 2 | 512B (8 cache lines) | 32 zmm 充裕，prefetch 两行 |
| NEON | 1 | 128B (2 cache lines) | 寄存器压力类似 AVX2 |
| SVE2 | 2 | 按 sve_vl_bytes 动态 | 可变长向量，按实际宽度计算 |
| GPU sm_80 | 2-3 | cp.async 128B 多条 | 异步拷贝不占计算单元 |
| GPU sm_90 | ∞ | TMA 单条指令 | TMA 硬件自动管理 prefetch，无需手动 |
| GPU sm_100+ | ∞ | TMEM 替代 SMEM | TMEM 直接做累加，无需 SMEM staging |

**预取距离推导公式**:

```
pf_distance = max(cache_line_bytes × depth, hidden_size × sizeof(elem))
```

### 13.2 FMA/Load 比率与微内核选择

微内核的 MR×NR 选择由寄存器预算和 FMA/Load 比率共同决定：

```
FMA_to_Load_Ratio = (MR × NR_vecs) / (1 + NR_vecs)
                   = MR × NR_vecs / (1 + NR_vecs)

NR_vecs = ceil(NR / simd_width)
accumulator_regs = MR × NR_vecs
scratch_regs = num_simd_regs - accumulator_regs
```

| 微内核 | FMA ops/iter | Load ops/iter | 比率 | 寄存器分配 |
|-------|------------|------------|------|----------|
| AVX2 6×16 | 6 | 2 | 3:1 | 12 累加器 + 1 A ptr + 1 B ptr + 2 scratch = 16 ymm |
| AVX-512 14×32 | 14 | 2 | 7:1 | 28 累加器 + 2 ptr + 2 scratch = 32 zmm |
| AMX 16×16 | 1 TDP | 2 TLOAD | 0.5:1 | 8 TMM tile，但单指令处理 16×16×16 = 4096 MAC |
| NEON 8×12 | 8 | 2 | 4:1 | 24 累加器 + 4 ptr + 4 scratch = 32 v |
| GPU sm_80 warp | 16×8×16 | — | — | 寄存器由 warp 分配，32 regs/thread |

**选择约束**: `accumulator_regs + min_scratch(2) + pointer_regs(2) ≤ num_simd_regs`

当 `scratch_regs < min_epilogue_regs` 时，epilogue 溢出到栈 → 性能断崖。这就是为什么 AVX2 只能 2-3 ops epilogue。

### 13.3 Epilogue 寄存器预算公式

```rust
/// 计算给定微内核规格下的最大 epilogue 操作数
fn max_epilogue_depth(num_simd_regs: usize, mr: usize, nr_vecs: usize) -> usize {
    let accumulator = mr * nr_vecs;
    let pointer_regs = 2;  // A ptr + B ptr
    let available = num_simd_regs - accumulator - pointer_regs;

    // 每个 TraceOp 需要 1-2 scratch 寄存器
    // Binary op (Add/Mul) = 2 regs, Unary op (Exp/Sqrt) = 1 reg
    let avg_regs_per_op = 1.5;
    (available as f32 / avg_regs_per_op).floor() as usize
}
```

| 硬件 | 总寄存器 | 累加器 | 指针 | 可用 scratch | 最大 epilogue |
|------|---------|--------|------|------------|-------------|
| AVX2 | 16 ymm | 12 | 2 | 2 | 1-2 ops |
| AVX-512 | 32 zmm | 28 | 2 | 2 | 1-2 ops（保守） |
| AVX-512 + NR 缩减 | 32 zmm | 14 | 2 | 16 | 6-8 ops |
| NEON | 32 v | 24 | 2 | 6 | 3-4 ops |
| NEON + NR 缩减 | 32 v | 16 | 2 | 14 | 6-8 ops |

**NR 缩减策略**: 当 epilogue 深度需求高时，Phase 2 可选择 `NR_variant < NR_nominal`，牺牲微内核吞吐换取更深 epilogue。trade-off 由 `FusionEngine` 根据 fusion plan 自动计算。

### 13.4 缓存层级资源分配算法

三级缓存资源分配模型：

```
L1 Cache (最快, 最小):
  ┌─────────────────────────────────┐
  │  GEMM A tile (MC × KC)          │  ← pack_a 输出
  │  GEMM B tile (KC × NC)          │  ← pack_b 输出
  │  [可选] Normed tile (MC × hidden)│  ← TileLevelFusion scratch
  └─────────────────────────────────┘
  约束: MC × KC × elem_bytes + KC × NC × elem_bytes ≤ L1 × 75%

L2 Cache:
  ┌─────────────────────────────────┐
  │  权重预取窗口 (KC × NC tiles)    │  ← software pipelining
  │  KV Cache 热页 (当前层)          │  ← Attention 访问
  │  激活值缓存 (hidden × MC)        │  ← 前驱输出
  └─────────────────────────────────┘
  分配策略:
    kv_budget   = L2 × 0.40  (Attention-bound 场景)
    weight_budget = L2 × 0.35 (GEMM-bound 场景)
    activation   = L2 × 0.25 (Epilogue scratch)

L3 Cache / HBM:
  ┌─────────────────────────────────┐
  │  全部模型权重                    │  ← 常驻
  │  全量 KV Cache (所有层)          │  ← PagedAttention 管理
  │  临时 buffer (Epilogue 输出)     │  ← 区间图着色复用
  └─────────────────────────────────┘
  分配策略:
    model_weights = model.param_count × dtype_size
    kv_cache      = total_pages × page_size × 2 × num_layers × kv_dim × dtype_size
    workspace     = min(activation_bytes, L3 × 30%)
```

**GPU 显存预算分配**:

```
total_hbm = DeviceProfile.total_memory
model_weights = sum(all_tensors) × quant_type_size_ratio  // 4-bit = 0.25, FP16 = 0.5
kv_pool       = total_hbm × 0.60 - model_weights
workspace     = total_hbm × 0.10  // 临时 buffer
reserved      = total_hbm × 0.05  // CUDA/驱动保留

max_pages = kv_pool / (page_size × 2 × num_layers × kv_dim × dtype_size)
```

### 13.5 新硬件特性积极使用策略

硬件特性不止是"支持"，而是**主动利用**——在检测到新特性时，JIT 编译器自动切换到更优路径：

#### GPU Blackwell (sm_100+)

| 特性 | 传统路径 | 积极利用路径 | 收益 |
|------|---------|------------|------|
| FP4 原生 | W4A8 量化 → 解量化 → F16 GEMM | 权重全程 FP4，GEMM 直接消费 FP4 | 显存 -50%，带宽 -50% |
| TMEM (256KB/SM) | SMEM 做 attention tiling | TMEM 替代 SMEM 做 attention + 累加 | shared memory 解放给 routing/compact |
| tcgen05.mma | mma.sync 16×8×16 | block-scaled GEMM with per-block scale | 消除独立 Scale 节点 |
| Thread Block Cluster | 单 CTA GEMM | 跨 CTA 协同 MMA | 2-CTA 协同扩展 tile 尺寸 |

#### GPU Hopper (sm_90)

| 特性 | 传统路径 | 积极利用路径 | 收益 |
|------|---------|------------|------|
| TMA 2D/5D | cp.async 手动预取 | TMA 单条指令异步预取 | 释放寄存器，减少指令数 |
| WGMMA | mma.sync warp 级 | warp-group 矩阵乘累加 | 4× 矩阵吞吐 |
| Warp Specialization | 同步 kernel | producer/consumer 分离 | 预取与计算完全重叠 |
| cuda::barrier | bar.sync 全局 | per-warp barrier | 细粒度同步减少等待 |

#### CPU AMX

| 特性 | 传统路径 | 积极利用路径 | 收益 |
|------|---------|------------|------|
| TDPBF16PS | BLIS FMA 微内核 | tile 8×8 BF16 矩阵乘 | GEMM 吞吐 2-4× |
| tile_config | 运行时固定 | per-layer 动态配置 tile 尺寸 | Attention 用小 tile，FFN 用大 tile |
| tile_release | 编译时持有 | FFN 完成后释放 tile 资源 | 释放后 zmm 全给 epilogue |

#### ARM SME2

| 特性 | 传统路径 | 积极利用路径 | 收益 |
|------|---------|------------|------|
| ZA Array | NEON 向量化 | 2D 矩阵存储 + outer product | 消除显式 tile 管理 |
| streaming SVE | NEON 固定 128-bit | 可变长 streaming mode | 矩阵专用向量单元 |
| SMOP (outer product) | NEON FMLA 累加 | 单指令 outer product | 矩阵乘效率大幅提升 |

**检测→切换流程**:

```
DeviceProfile::detect()
  → has_amx=true → GemmStrategy::AmxTile (替代 BLIS)
  → sm_version=90 → FlashAttention::FA3Pipeline (替代 FA2Tiled)
  → sm_version=100 → FlashAttention::FA4BlockScaled (替代 FA3)
  → has_sme2 → GemmStrategy::SmeOuterProduct (替代 NEON BLIS)
```

## 12. 交叉引用

| 主题 | 位置 |
|------|------|
| JIT 四阶段管线 | `SPEC/01-JIT-PIPELINE.md` |
| 硬件探测与 DeviceProfile | `SPEC/02-HARDWARE.md` |
| 计算图 IR | `SPEC/03-GRAPH-IR.md` |
| 核心铁律 | `SPEC/00-PHILOSOPHY.md` |
| 运行时优化 | `SPEC/05-OPTIMIZATIONS.md` |
| 支持模型清单 | `SPEC/11-MODELS.md` |
