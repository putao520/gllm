# gllm 原生模型权重分发格式 (GLLM-WEIGHT-FORMAT)

> **实现状态**: ✅ REQ-GLF-001~007 全部完成; RTN 量化 baseline 已实现 (AWQ4/GPTQ4/NVFP4)，AWQ/GPTQ activation-aware 校准为未来增强
>
> **SSOT**: 本文件定义 gllm 原生模型权重分发格式 `.gllm`。权重在文件中即以量化格式存储（AWQ4/GPTQ4/NVFP4/FP8 等），推理时通过分页机制按需加载到 GPU，由 GEMM prologue（QuantGather）on-the-fly 解量化后送入 Tensor Core。
>
> 交叉引用: `35-QUANTIZED-WEIGHT-PAGE-COMPRESSION.md`（量化权重页架构）、`23-QUANT-CODEGEN-ALGO.md`（QuantFormatDescriptor 22 种量化格式）、`21-WEIGHT-PAGING.md`（权重分页 §8 量化权重页）、`07-LOADER.md`（模型加载规范）

## §0 设计原则

1. **量化原生存储**: 权重在文件中即以量化格式存储，加载时零解压，GEMM prologue 负责解量化
2. **分页友好**: 文件布局与 GPU 分页对齐，支持 mmap + 按需加载
3. **自描述格式**: 内嵌模型架构元数据，无需外部 config.json
4. **量化格式覆盖**: 支持 SPEC 23 定义的 22 种 QuantType
5. **向后兼容**: 提供从 GGUF/safetensors 到 `.gllm` 的离线转换工具

## §1 文件格式

```
GLLM Weight File (.gllm):
┌──────────────────────────────────────┐
│ Header (64 bytes):                    │
│   magic:          "GLLM" (4B)         │
│   version:        u32 = 1             │
│   flags:          u32 (bit 0: 量化)   │
│   meta_offset:    u64                 │
│   tensor_count:   u32                 │
│   tensor_dir_offset: u64              │
│   data_offset:    u64                 │
│   page_size:      u32 (分页对齐)       │
│   reserved:       [u8; 28]            │
├──────────────────────────────────────┤
│ Tensor Directory:                     │
│   per tensor (64 bytes):              │
│     name_offset:      u32             │
│     name_len:         u16             │
│     ndim:             u8              │
│     dtype:            u8 (DType)      │
│     shape:            [u64; 4]        │
│     quant_format:     u8 (QuantType)  │
│     quant_block_size: u16             │
│     scale_dtype:      u8 (DType)      │
│     zp_type:          u8 (0=无/1=u8)  │
│     data_offset:      u64             │
│     compressed_size:  u64             │
│     original_size:    u64             │
├──────────────────────────────────────┤
│ String Table:                         │
│   tensor names (null-terminated)      │
├──────────────────────────────────────┤
│ Metadata (MessagePack):               │
│   model_type, arch_key, vocab_size,   │
│   hidden_size, num_layers, num_heads, │
│   head_dim, intermediate_size, ...    │
├──────────────────────────────────────┤
│ Quantized Tensor Data:                │
│   For each tensor (page-aligned):     │
│     packed_quant_data: [u8; ...]      │
│     scales: [scale_dtype; n_blocks]   │
│     zero_points: [u8; ...] (可选)     │
│     (padding to page_size boundary)   │
└──────────────────────────────────────┘
```

### §1.1 Header 字段说明

| 字段 | 偏移 | 大小 | 说明 |
|------|------|------|------|
| `magic` | 0 | 4 | "GLLM" ASCII |
| `version` | 4 | 4 | 格式版本 (当前 1) |
| `flags` | 8 | 4 | bit 0: 全局量化标志 |
| `meta_offset` | 12 | 8 | MessagePack 元数据偏移 |
| `tensor_count` | 20 | 4 | 张量数量 |
| `tensor_dir_offset` | 24 | 8 | Tensor Directory 偏移 |
| `data_offset` | 32 | 8 | 数据区偏移 |
| `page_size` | 40 | 4 | 分页对齐大小 (通常 4096) |
| `reserved` | 44 | 20 | 保留 |

### §1.2 Tensor Directory 条目

每个张量 64 字节。`quant_format` 字段对应 SPEC 23 `QuantType` 枚举（AWQ4/GPTQ4/NVFP4/FP8 等）。

当 `quant_format = 0`（QuantType::None）时，数据为原始精度（FP32/BF16/FP16），`compressed_size = original_size`。

当 `quant_format != 0` 时：
- `compressed_size` = 量化后字节数（packed data + scales + zero_points）
- `quant_block_size` = 量化块大小（如 128 for AWQ4, 32 for NVFP4）
- `scale_dtype` = 缩放因子精度
- `zp_type` = 零点类型（0 = 无零点，如 NVFP4；1 = u8 零点，如 GPTQ4）

### §1.3 量化数据块布局

每个张量的量化数据按 `quant_block_size` 分块存储，与 `QuantBlockLoad` VmInstr (SPEC 26 §1.2.1) 对齐：

```
AWQ4 布局 (行优先, block_size=128):
  per block:
    packed_weights: [u4; 128] = 64 bytes (行优先打包)
    scale: f16 = 2 bytes
    zero_point: f16 = 2 bytes (可选)
  total per block: 68 bytes (128 元素, 压缩比 7.5:1 vs FP32)

GPTQ4 布局 (列交织, block_size=128):
  per block:
    packed_weights: [u4; 128] = 64 bytes (列交织打包)
    scale: f16 = 2 bytes
    zero_point: i32 packed = 4 bytes (INT4 packed + 1 偏移)
  total per block: 70 bytes

NVFP4 布局 (两级缩放, block_size=16, sub_block=16):
  per 128 elements (8 blocks × 16 sub_blocks):
    packed_weights: [E2M1; 128] = 64 bytes
    sub_block_scales: [UE4M3; 8] = 8 bytes
    global_scale: f16 = 2 bytes (per 128-element tile)
  total per 128 elements: 74 bytes

FP8 布局 (无块结构):
  per element: 1 byte (E4M3 或 E5M2)
  无 scales, 无 zero_points
  total: n_elements bytes (4:1 vs FP32)
```

## §2 与分页系统的集成

```
.gllm 文件加载流程:
  1. mmap 整个文件到虚拟地址空间
  2. 解析 Header + Tensor Directory + Metadata
  3. 构建 ModelArchKey + TensorMap
  4. 从 Metadata 推导模型架构 (auto_graph 所需参数)
  5. 对于每个权重 tensor:
     a. 从 quant_format 和 block_size 计算页内块数
     b. 创建 QuantizedWeightPage 映射 (file_offset → page_id)
     c. 注册到 UnifiedVirtualPage 系统 (SPEC 21 §8)
  6. 推理时按需加载:
     a. 缺页 → 从 mmap 读取量化数据块
     b. DMA 到 GPU HBM
     c. QuantBlockLoad + QuantGather 解量化 → GEMM
```

**零解压加载**: 文件中的数据就是量化格式，不需要任何 CPU 端解压或解量化。GPU 端由 GEMM prologue（QuantGather）on-the-fly 处理。

## §3 转换工具

### §3.1 GGUF → .gllm

```bash
gllm-convert --input model.gguf --output model.gllm \
    --quant-format awq4 --block-size 128 --calibration-data wiki.jsonl
```

转换流程:
1. 解析 GGUF header + tensor metadata
2. 如果 GGUF 已包含量化权重 (Q4_0/Q5_1 等):
   a. 读取量化参数 (block_size, n_blocks)
   b. 转换为 .gllm 原生量化布局（repack 如果布局不同）
   c. 直接写入 packed data + scales
3. 如果 GGUF 包含 FP16/BF32 权重:
   a. 读取原始权重
   b. 使用 calibration data 执行量化校准
   c. 按 quant_format 编码 (AWQ4/GPTQ4/NVFP4)
   d. 写入 packed data + scales + zero_points
4. 构建 Tensor Directory + Metadata
5. 写入 .gllm 文件

### §3.2 safetensors → .gllm

```bash
gllm-convert --input model.safetensors --output model.gllm \
    --quant-format nvfp4 --calibration-data wiki.jsonl
```

safetensors 中权重为原始精度（BF16/FP16/F32），必须执行量化校准：
1. 加载原始权重
2. 运行 calibration data 收集激活统计
3. 计算最优 per-block scale/zp（AWQ: activation-aware; GPTQ: Hessian-based）
4. 量化编码 + 写入 .gllm

### §3.3 GGUF 已量化 → .gllm（直通模式）

对于已经是量化格式的 GGUF（如 Q4_0、Q8_0），可以跳过重新量化：

```bash
gllm-convert --input model-Q4_0.gguf --output model.gllm --passthrough
```

此模式仅 repack GGUF 的量化布局到 .gllm 格式，不改变精度。

## §4 模型架构自动识别

从 Tensor Directory + Metadata 自动推导 `ModelArchKey`，无需外部配置文件：

```
识别流程:
  1. 读取 Metadata 中的 arch_key (如 "llama", "qwen3", "gemma4")
  2. 读取 Metadata 中的关键参数 (hidden_size, num_layers, num_heads, ...)
  3. 从 Tensor Directory 的张量名模式确认架构特征
     (如 "model.layers.*.self_attn.q_proj" → GQA 架构)
  4. 构建 ModelArchKey → auto_graph 直接构建 CompilerGraph
  5. 从 Tensor Directory 的 quant_format 为每个权重张量配置 QuantType
```

**与现有 auto_graph 的兼容**: `.gllm` 的 Metadata 包含 GGUF equivalent 的所有架构参数。`auto_graph` 无需修改，只需在 Loader 层增加 `.gllm` 格式解析 + QuantType 透传。

## §5 压缩比预估

| 模型 | 原始 (FP16) | INT4 量化 | NVFP4 量化 | FP8 量化 |
|------|-----------|----------|-----------|---------|
| Qwen3-8B | ~16 GB | ~4.5 GB (3.6x) | ~4 GB (4x) | ~8 GB (2x) |
| Llama-4-17B | ~34 GB | ~9.5 GB (3.6x) | ~8.5 GB (4x) | ~17 GB (2x) |
| Gemma-4-31B | ~62 GB | ~17 GB (3.6x) | ~15.5 GB (4x) | ~31 GB (2x) |
| DeepSeek-V3-671B | ~1.3 TB | ~365 GB (3.6x) | ~325 GB (4x) | ~650 GB (2x) |

**说明**: INT4 量化包含 per-block scale/zp 开销（约 6-10%），因此实际压缩比略低于理论的 4:1。

## §6 禁止规则

- ❌ 禁止在加载时执行 CPU 侧解量化 — 量化数据直接映射到 GPU
- ❌ 禁止将 FP32/FP16 权重写入 `.gllm`（除非 quant_format=None 的 embed 层）
- ❌ 禁止硬编码量化格式 — 由 Tensor Directory 的 quant_format 字段驱动
- ❌ 禁止依赖外部 config.json — 所有元数据内嵌
- ❌ 禁止使用函数逼近压缩（FWT/SIREN/INR/F-INR）— 实测不可行

## §7 REQ 清单

| REQ ID | 描述 | 验收标准 | 依赖 |
|--------|------|---------|------|
| REQ-GLF-001 | .gllm 文件格式定义 | Header + TensorDirectory + StringTable + Metadata + QuantizedData 布局完整 | 无 |
| REQ-GLF-002 | 量化数据块布局 | packed_weights + scales + zero_points 按 QuantType 规范存储，page-aligned | SPEC 23 |
| REQ-GLF-003 | 分页 mmap 加载器 | mmap .gllm → QuantizedWeightPage 映射 → UnifiedVirtualPage 注册 | REQ-GLF-001, SPEC 21 |
| REQ-GLF-004 | GGUF → .gllm 离线转换器 (已量化) | Q4_0/Q5_1 等直通 repack，推理结果与原 GGUF 余弦相似度 > 0.999 | REQ-GLF-001, REQ-GLF-002 |
| REQ-GLF-005 | safetensors → .gllm 离线转换器 | BF16/FP16 + calibration → AWQ4/GPTQ4/NVFP4，perplexity 损失 < 0.1 | REQ-GLF-001, REQ-GLF-002 |
| REQ-GLF-006 | 模型架构自动识别 | 从 TensorDir + Metadata 推导 ModelArchKey，auto_graph 可直接构建图 | REQ-GLF-001 |
| REQ-GLF-007 | GGUF → .gllm 转换器 (原始精度) | FP16/BF16 权重 + calibration → 量化编码，perplexity 损失 < 0.1 | REQ-GLF-005 |
