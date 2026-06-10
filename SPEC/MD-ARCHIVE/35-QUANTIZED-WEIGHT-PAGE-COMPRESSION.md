# 量化原生权重页压缩架构 (QUANTIZED-WEIGHT-PAGE-COMPRESSION)

> **实现状态**: ✅ 全部完成 — REQ-QWP-001~007 (QuantBlockLoad + QuantGather GEMM prologue + .gllm 量化加载 + GGUF 转换 + 权重页 Tier 透传 + REQ-QWP-005 QuantGather 双缓冲 + REQ-QWP-006 SM61 解量化路径)
>
> **SSOT**: 本文件定义量化权重页的压缩架构 — 量化即权重压缩的唯一可行路径。
>
> **实验依据**: FWT 波函数反推、SIREN/INR、Hash Grid + Gabor、F-INR 泛函张量分解四种方法在真实 LLM 权重上实测均失败（cos_sim < 0.02 ~ 0.81，需 77%+ 参数才能达到 0.99）。根因：LLM 权重是高熵（近似 i.i.d. 高斯）数据，信息论限制任何函数逼近方法的压缩比上限约为 1:1。

<div data-cross-repo-xrefs>
<b>跨仓库依赖 (gllm-nccl)</b>:
量化权重页传输与通信管线量化协同:
<a data-xref-id="REQ-DP-008" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-008">REQ-DP-008</a>
(带量化压缩的 RDMA 页传输) 消费本文件 QuantGather 解量化 + gllm-nccl <a data-xref-id="REQ-QUANT-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-QUANT-001">REQ-QUANT-001</a>
<a data-xref-id="REQ-QUANT-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-QUANT-002">REQ-QUANT-002</a> (GPU 原生量化模板 + 量化通信管线) |
权重页 Tier 透传 (HBM/DRAM/NVMe) 与 <a data-xref-id="REQ-DP-004" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-004">REQ-DP-004</a>
(迁移计划生成) 协同
</div>
>
> **量化可行的根因**: 量化不是"把数据变成公式"，而是"降低每个数值的精度"。4-bit 量化 = 8x 压缩（从 FP32），perplexity 损失 < 0.1。这是有损压缩，但损失的精度不影响推理质量。
>
> 交叉引用: `23-QUANT-CODEGEN-ALGO.md`（QuantFormatDescriptor 22 种量化格式）、`24-QUANT-PIPELINE-JIT.md`（QuantGather/QuantGemm JIT 管线）、`22-PAGE-COMPRESSION.md`（CompressionCodec，权重页 codec=None）、`21-WEIGHT-PAGING.md`（量化权重页 §8）、`36-GLLM-WEIGHT-FORMAT.md`（.gllm 量化权重格式）

## §0 设计原则

1. **量化即压缩**: 权重压缩的唯一路径是量化（4-8 bit），不是函数逼近
2. **解量化即 GEMM prologue**: 量化权重的"解压"与 GEMM 融合（QuantGather），不是独立 pass
3. **页级透传**: 量化权重页以量化格式直接存储和传输，GEMM JIT 内 on-the-fly 解量化
4. **格式覆盖**: 支持 SPEC 23 定义的 22 种 QuantType（NVFP4/AWQ4/GPTQ4/SqueezeLLM/FP8 等）
5. **与现有管线完全复用**: QuantGather/QuantGemm 已在 SPEC 24 定义，本文件仅定义分页集成

### §0.1 为何函数逼近不可行

四种方法实测数据（SmolLM2-135M q_proj 576×576）：

| 方法 | 最佳 cos_sim | param_ratio | 测试脚本 |
|------|------------|-------------|---------|
| FWT 波函数反推 | K=16: ~0.10 | 73% 系数才到 99% 能量 | `scripts/fwt_real_weight_test.py` |
| SIREN/INR | 0.99+ | ~77% | `scripts/siren_weight_compression_test.py` |
| Hash Grid + Gabor | 0.81 | 81% | `scripts/hashgrid_gabor_test.py` |
| F-INR 泛函张量分解 | 0.01 | <0.5% | `scripts/finr_weight_compression_test.py` |

**根因**: LLM 权重近似 i.i.d. 高斯分布，信息熵极高。1 bit 信息需要至少 1 bit 存储，任何参数化函数逼近方法都无法突破此限制。

**量化为何不受此限**: 量化降低每个数值的精度（FP32→INT4 = 8x 压缩），是逐元素操作，不依赖数据的低秩/低熵假设。4-8 bit 量化在所有主流 LLM 上已被验证 perplexity 损失 < 0.1。

## §1 量化权重页架构

### §1.1 端到端数据流

```
离线阶段 (gllm-convert 工具):
  原始权重 (FP32/BF16/FP16)
    → 量化校准 (calibration data → optimal scale/zp per block)
    → 量化编码 (FP32 → AWQ4/GPTQ4/NVFP4/FP8/...)
    → 写入 .gllm 文件 (SPEC 36)

推理阶段 (GPU HBM 内):
  量化权重页 (从 .gllm mmap 或 weight paging 加载)
    → QuantBlockLoad (SPEC 26 VmInstr, 加载 packed data + scales + zp)
    → QuantGather (SPEC 24, GEMM prologue, on-the-fly 解量化)
    → 解量化后权重驻留寄存器
    → Tensor Core GEMM (累加器 F32)
    → Epilogue 写回 (可选: 窄化到原始 dtype)
```

### §1.2 量化格式 → 压缩比映射

| 量化格式 | bits/elem | 压缩比 (vs FP32) | 压缩比 (vs FP16) | perplexity 损失 |
|---------|----------|-----------------|-----------------|----------------|
| FP8 (E4M3/E5M2) | 8 | 4:1 | 2:1 | < 0.01 |
| NVFP4 (E2M1+UE4M3) | 4+scale | ~7:1 | ~3.5:1 | < 0.05 |
| AWQ4 (行优先+FP16 zp) | 4+zp | ~7:1 | ~3.5:1 | < 0.1 |
| GPTQ4 (列交织+INT4 zp) | 4+zp | ~7:1 | ~3.5:1 | < 0.1 |
| INT4 (均匀量化) | 4 | 8:1 | 4:1 | < 0.3 |
| SqueezeLLM (3-bit LUT) | 3+codebook | ~9:1 | ~4.5:1 | < 0.2 |
| MXFP4 (E2M1+E8M0) | 4+shared_exp | ~7:1 | ~3.5:1 | < 0.1 |

### §1.3 与 SPEC 22 CompressionCodec 的关系

权重页的 `CompressionCodec` 固定为 `None = 0`。

量化权重页的"压缩"不是通过 CompressionCodec 实现的，而是通过 **QuantType** 实现的：
- `CompressionCodec` 管页级字节流压缩（LZ4/BitPackRle/ZstdDict）— 用于 KV Cache
- `QuantType` 管权重级精度压缩（AWQ4/GPTQ4/NVFP4）— 用于权重页
- 两者正交：量化权重页 = QuantType 压缩 + CompressionCodec=None（不需要额外字节流压缩）

## §2 GEMM Prologue 解量化融合

### §2.1 QuantGather 解量化路径

量化权重页的解量化由 QuantGather VmInstr（SPEC 24 §2.2）在 GEMM prologue 中完成：

```
QuantGather 数据流:
  1. QuantBlockLoad: 从 Global Memory 加载 packed_weights + scales + zp
     → 按 QuantFormatDescriptor 的 block_size 分块
     → cp.async 异步加载到 Shared Memory
  2. On-the-fly 解量化:
     → 每个线程处理一个 block
     → dequant = (packed_weight - zp) × scale
     → 解量化结果直接写入寄存器 (Fragment A/B)
  3. GEMM Compute:
     → Tensor Core MMA 指令消费寄存器中的解量化数据
     → 累加器 F32
```

### §2.2 硬件特化路径

| SM 版本 | 解量化路径 | 优势 |
|---------|-----------|------|
| SM100 (Blackwell) | tcgen05 原生 FP4 解量化 | 硬件级，零延迟 |
| SM90 (Hopper) | WGMMA + 寄存器内解量化 | TMA 异步加载 + 解量化与 GEMM 重叠 |
| SM80 (Ampere) | mma.sync + Shared Memory staging | cp.async + 寄存器内解量化 |
| CPU (AVX-512) | VNNI + VPDPBUSD + scale broadcast | 4-bit unpack + FMA 融合 |

### §2.3 与 Mega-Kernel 的集成

量化权重解量化集成到 Mega-Kernel 的 device function 集合中（SPEC 32）：
- `prefill_device_fn`: QuantGather + FlashAttention GEMM（解量化是 GEMM prologue）
- `decode_device_fn`: QuantGather + decode GEMV（解量化是 GEMV prologue）

**双缓冲优化**:
```
时间线:
  t0: [QuantGather 解量化 Page 0 → Fragment A] [GEMM 使用旧数据]
  t1: [GEMM 消费 Fragment A]                   [QuantGather 解量化 Page 1 → Fragment B]
  t2: [GEMM 消费 Fragment B]                   [QuantGather 解量化 Page 2 → Fragment A]
```

## §3 量化权重页与分页系统

### §3.1 页内布局

量化权重页的物理布局必须与 `QuantBlockLoad` VmInstr（SPEC 26 §1.2.1）的加载模式对齐：

```
QuantizedWeightPage (page-aligned):
  ┌────────────────────────────────────────┐
  │ QuantBlock 数据 (packed_weights):      │
  │   按 block_size 分块，每块包含:         │
  │   - packed quantized values (4/8 bit)  │
  │   - scale factor (FP16/FP32/UE4M3)    │
  │   - zero_point (可选)                  │
  │   布局: 行优先 (AWQ) 或 列交织 (GPTQ)  │
  │   或 两级缩放 (NVFP4: E2M1+UE4M3)     │
  ├────────────────────────────────────────┤
  │ Padding 到 page_size 对齐              │
  └────────────────────────────────────────┘
```

**布局对齐规则**:
- 页大小 = `QuantFormatDescriptor.block_size` 的整数倍
- packed_weights 连续存储，scales 紧跟其后（或按格式定义的布局）
- 页边界必须对齐到 CUDA 内存事务大小（128B for SM80+, 256B for TMA）

### §3.2 权重页生命周期

```
加载时:
  Loader 读取 .gllm Tensor Directory
    → 识别 quant_format + shape + data_offset
    → 按 block_size 计算页数
    → 创建 QuantizedWeightPage 映射
    → 注册到 UnifiedVirtualPage 系统

推理时 (按需加载):
  GEMM 需要权重 tile
    → 检查 WeightPage 是否在 HBM
    → 如果不在: 从 .gllm mmap 读取量化数据 → DMA 到 HBM
    → QuantBlockLoad 加载 packed data + scales
    → QuantGather on-the-fly 解量化 → GEMM
```

### §3.3 与 WeightTier 的协同

| Tier | 存储内容 | 加载到 HBM 后 |
|------|---------|-------------|
| DeviceLocal (HBM) | 量化权重页 (packed + scales) | QuantBlockLoad → QuantGather → GEMM |
| HostLocal (DRAM) | 量化权重页 (同上) | DMA 到 HBM → 同上 |
| DiskMmap (NVMe) | .gllm 文件中的量化数据 | pread → DMA 到 HBM → 同上 |

**关键**: 权重在任何 Tier 都以量化格式存储。Tier 切换只是物理位置变化，不需要解量化/重量化。

## §4 与 SPEC 36 .gllm 格式的集成

.gllm 文件（SPEC 36）是量化权重的原生存储格式：

```
.gllm 文件中量化权重存储:
  Tensor Directory 条目:
    compression: None (不使用 CompressionCodec)
    quant_format: QuantType (AWQ4/GPTQ4/NVFP4/FP8/...)
    quant_block_size: u32 (量化块大小)
    data_offset: u64 (指向 packed data)
    compressed_size: u64 (量化后字节数)
    original_size: u64 (原始 FP32 字节数)

  数据区 (page-aligned):
    packed_quant_data: [u8; ...]   // 量化后权重
    scales: [scale_dtype; ...]     // per-block 缩放因子
    zero_points: [u8; ...]         // per-block 零点 (可选)
```

加载时零解压：量化数据直接映射为 QuantizedWeightPage，GEMM prologue 处理解量化。

## §5 禁止规则

- ❌ 禁止对权重页使用函数逼近压缩（FWT/SIREN/INR/Hash Grid/F-INR）— 实测不可行
- ❌ 禁止对权重页使用 CompressionCodec（LZ4/BitPackRle/ZstdDict）— 量化格式已是压缩态
- ❌ 禁止独立的权重解量化 pass — 必须与 GEMM 融合（QuantGather prologue）
- ❌ 禁止在 CPU 侧解量化权重 — 必须在 GPU JIT 内完成
- ❌ 禁止硬编码量化格式 — 由 .gllm Tensor Directory 的 quant_format 字段驱动
- ❌ 禁止量化格式不支持时 fallback 到 FP32 — 必须返回错误

## §6 REQ 清单

| REQ ID | 描述 | 验收标准 | 依赖 |
|--------|------|---------|------|
| REQ-QWP-001 | 量化权重页布局定义 | QuantizedWeightPage header + data 与 QuantBlockLoad VmInstr 对齐，page-aligned | SPEC 23 |
| REQ-QWP-002 | GEMM prologue 解量化集成 | QuantGather 从量化权重页解量化 + GEMM 融合，数值精度与 SPEC 10 一致 | SPEC 24 |
| REQ-QWP-003 | .gllm 量化权重加载 | 从 .gllm 文件加载量化权重页，推理结果与原始格式余弦相似度 > 0.999 | SPEC 36 |
| REQ-QWP-004 | 权重页 Tier 透传 | 量化权重在 HBM/DRAM/NVMe 间切换无需解量化/重量化，数据格式不变 | SPEC 21 |
| REQ-QWP-005 | 双缓冲 QuantGather 流水线 | 解量化与 GEMM 重叠，吞吐量 > 单缓冲的 1.5× | REQ-QWP-002 |
| REQ-QWP-006 | 硬件特化解量化路径 | SM80/SM90/SM100/CPU 四路径通过 DeviceProfile 驱动选择 | SPEC 24, SPEC 17 |
| REQ-QWP-007 | GGUF 量化权重 → .gllm 转换 | 支持 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 等格式转换为 .gllm 原生量化格式 | SPEC 36 |
