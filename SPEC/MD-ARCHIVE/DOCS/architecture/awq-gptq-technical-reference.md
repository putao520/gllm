# AWQ/GPTQ 技术参考 (Activation-Aware / Hessian-Based Post-Training Quantization)

> **用途**: 本文档为 gllm 理解 AWQ/GPTQ 量化格式提供底层技术参考。
> 涵盖物理存储布局 (交织 vs 行优先)、zero-point 形态差异、反量化公式、与 GGUF 生态的集成方式。
> 实现规范见 `SPEC/23-QUANT-CODEGEN-ALGO.md`。

## §1 概述

| 格式 | 核心算法 | 量化目标 | 来源生态 |
|------|---------|---------|---------|
| **AWQ** | Activation-aware: 按 activation magnitude 保护 ~1% salient weights, 仅量化非重要权重 | Weight-only 4-bit | AutoAWQ / vLLM (CUDA/Triton) |
| **GPTQ** | Hessian 近似: Oblique dog 逐列量化, 补偿量化误差到未量化列 | Weight-only 4-bit | AutoGPTQ / ExLlamaV2 (CUDA) |

**关键共性**: 两者反量化公式数学上完全一致, 差异仅在物理存储打包顺序和 zero-point 处理。

## §2 反量化数学公式

对任意已解包为整数 (0-15) 的 4-bit 权重 `qw`:

```
FP16_weight = (qw - zero_point) × scale
```

| 参数 | AWQ | GPTQ |
|------|-----|------|
| `scale` | FP16, 每 group_size=128 权重共享一个 | FP16, 每 group_size=128 权重共享一个 |
| `zero_point` | FP16, 直接从张量读取 | INT4 打包在 u32 中, 需位移解包; 通常需 +1 偏移补偿 |
| `qw` 解包 | 行优先连续 nibble 解包 | 列交织 nibble 解包 |

## §3 物理存储布局

### §3.1 核心事实: 张量分离存储 (非紧凑 struct)

在实际 CUDA 算子 (AutoAWQ / ExLlamaV2) 中, qweight、scales、zeros 是三个**独立的张量 (Tensor)**, 不是 C struct 紧凑排列:

```
内存布局:

qweight_tensor: [u32; N]  — 打包的 4-bit 权重
scales_tensor:  [f16; M]  — 每 group 一个 FP16 scale
zeros_tensor:   [f16; M] (AWQ) 或 [u32; P] (GPTQ) — 每 group 一个 zero-point
```

gllm-kernels 当前定义的 `BlockAWQ4 { qweight: [u32; 32], scales: f16, zeros: f16 }` 紧凑 struct
与实际 AWQ/GPTQ 的张量分离存储不匹配, 需要修正。

### §3.2 AWQ — 行优先连续打包

每 8 个连续 4-bit 权重打包进一个 u32:

```
u32 payload:

[W0 W1 W2 W3 W4 W5 W6 W7]

Bit 3:0   = W0 (nibble)
Bit 7:4   = W1 (nibble)
Bit 11:8  = W2 (nibble)
Bit 15:12 = W3 (nibble)
Bit 19:16 = W4 (nibble)
Bit 23:20 = W5 (nibble)
Bit 27:24 = W6 (nibble)
Bit 31:28 = W7 (nibble)
```

逻辑大小: group_size=128 × 4-bit = 64 bytes = 16 个 u32。

Zero-point: 直接以 FP16 存储在独立张量中, 每 group 一个 FP16 值。

### §3.3 GPTQ — 列交织打包

为了 GPU Warp 内线程合并访存 (Coalesced Access), 每 8 个**非连续**权重打包进一个 u32:

```
u32 payload:

[W0 W16 W32 W48 W64 W80 W96 W112]

 stride = 16 (典型值, 对应 warp 内 32 线程各取一行)
```

这种交织使得一个 Warp 的 32 个线程可以同时访问同一 u32 的不同 nibble, 避免银行冲突。

Zero-point: 8 个 4-bit 整数 zero-point 同样打包进一个 u32:
```
u32 zeros_payload:

[ zp0 | zp1 | zp2 | zp3 | zp4 | zp5 | zp6 | zp7 ]

运行时: zp_i = (zeros >> (i * 4)) & 0xF
GPTQ 通常需要 +1 偏移: effective_zp = zp_i + 1
```

### §3.4 存储布局对比

| 特性 | AWQ | GPTQ |
|------|-----|------|
| 打包顺序 | 行优先连续 | 列交织 (stride-16) |
| 每 u32 包含 | 8 个连续权重 | 8 个间隔权重 |
| Scale 存储 | 独立 FP16 张量 | 独立 FP16 张量 |
| Zero-point 存储 | 独立 FP16 张量 (每 group 一个 f16) | 打包 INT4 张量 (8 个 zp 打包进 u32) |
| Zero-point 解包 | 直接读取 FP16 | 位移 + 掩码 → +1 偏移 |
| 访存模式 | 标量连续读 | Warp 合并访存 |

## §4 GPU 算子执行模型

### §4.1 Weight-Only 反量化 GEMM

AWQ/GPTQ 都是 **Weight-Only** 量化: 激活 (Activation) 保持 FP16, 仅权重压缩到 4-bit。

GEMM 执行流程:

```
1. 从 qweight_tensor 加载 u32 到寄存器
2. 位移 + AND 解包 4-bit nibble → 整数 (0-15)
3. 从 scales_tensor 加载 FP16 scale
4. 从 zeros_tensor 加载 zero-point (AWQ: 直接 FP16; GPTQ: 解包 +1)
5. 反量化: FP16_weight = (qw - zp) × scale  (在寄存器中完成)
6. 发射 mma.sync (FP16 × FP16 → FP32 累加)
```

**关键**: 反量化在寄存器中完成, 不写回显存。这是 AWQ/GPTQ GEMM kernel 的核心性能优化。

### §4.2 与硬件原生 NVFP4 的对比

| 维度 | AWQ/GPTQ | NVFP4 (SM100) |
|------|----------|---------------|
| 量化域 | 整数 (INT4) | 浮点 (E2M1) |
| 硬件加速 | 无 (软件反量化 → FP16 → HMMA) | 原生 (硬件 Tensor Core 内部解码) |
| Scale 类型 | FP16 per-group | FP8 per-sub-block + F32 global |
| Zero-point | 有 (AWQ: FP16; GPTQ: INT4 packed) | 无 |
| 反量化位置 | 寄存器 (多条算术指令) | Tensor Core 内部 (零额外指令) |
| 打包约束 | AWQ 行优先 / GPTQ 列交织 | 小端序连续 nibble |

## §5 与 GGUF/llama.cpp 生态的关系

### §5.1 不原生支持

llama.cpp / GGUF **不直接原生运行** AWQ 或 GPTQ 的原始张量格式。

生态割裂:
- AWQ/GPTQ 属于 Python/PyTorch 生态 (依赖 CUDA/Triton 算子: vLLM, AutoGPTQ, AutoAWQ)
- GGUF 属于 C/C++ 生态 (依赖 llama.cpp)

### §5.2 转换流程

```
AWQ/GPTQ safetensors (HuggingFace)
    ↓ convert_hf_to_gguf.py
    ↓ 读取 AWQ/GPTQ 权重 → 反量化到 FP16 → 重新量化为 GGUF 格式
    ↓
GGUF 文件 (Q4_K_M / Q4_0 / Q4_1 / ...)
```

转换过程中, llama.cpp 会:
1. 读取 AWQ/GPTQ 的权重
2. 用反量化公式 `(qw - zp) × scale` 还原为 FP16
3. 重新量化为 GGUF 自有格式 (如 Q4_0: 每 32 权重共享一个 FP16 scale, 无 zero-point, 默认 zp=8)

### §5.3 对 gllm 的影响

| 加载路径 | AWQ/GPTQ 格式处理 |
|---------|-------------------|
| GGUF 文件 | 不需要处理 AWQ/GPTQ 格式 — 已转换为 GGUF 自有格式 |
| HuggingFace safetensors (AWQ/GPTQ 模型) | 需要理解 AWQ/GPTQ 格式才能正确加载和解量化 |

## §6 gllm 实现注意事项

### §6.1 Block 结构体修正

当前 `BlockAWQ4` / `BlockGPTQ4` 的紧凑 struct 定义需要修正:
- 实际存储是张量分离的 (qweight/scales/zeros 各自独立)
- 紧凑 struct 会误导 Weight Loader 按 struct 偏移读取
- GGUF 路径下不会遇到原始 AWQ/GPTQ 格式, 但 safetensors 直加载路径会

### §6.2 QuantFormatDescriptor 注册修正

当前注册使用 `ScaleLayout::Hierarchical` (K-Quant 的 packed scale 布局), 不符合 AWQ/GPTQ 的线性 per-group scale/zero 布局。AWQ/GPTQ 的 scale 和 zero-point 是简单的每 group 一个标量, 不需要分层的 packed scale。

### §6.3 量化 GEMM 路径

AWQ/GPTQ 在 gllm 中的 GEMM 路径与标准 INT4 不同:
- 标准 INT4 (Q4_0/Q4_1): `DK::PackedInt4` → nibble 解包 → int8 dot-product → scale
- AWQ/GPTQ: 需要先反量化到 FP16 → 走 FP16/BF16 GEMM 路径 (Weight-Only 模式)
- 这与 SPEC/23 §1.1 "原生量化计算铁律" 不矛盾 — AWQ/GPTQ 本质是 Weight-Only 反量化到 FP16, 不是 "decode to F32 then FMA"

## §7 参考

- AWQ 论文: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., 2023)
- GPTQ 论文: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022)
- AutoAWQ: https://github.com/casper-hansen/AutoAWQ
- AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
- ExLlamaV2 (GPTQ 变体): https://github.com/turboderp/exllamav2
- llama.cpp convert_hf_to_gguf.py: AWQ/GPTQ → GGUF 转换逻辑
