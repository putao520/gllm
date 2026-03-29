# gllm 数据结构定义

> **📌 SSOT**: 本文档定义 gllm 项目中的核心数据结构。
> **关联需求**: REQ-MODEL-001 ~ REQ-MODEL-005, REQ-LOADER-001 ~ REQ-LOADER-021

## 1. 模型格式 (DATA-FORMAT)

### 1.1 支持的模型格式

| 格式 | 文件扩展名 | 主要用途 | 状态 |
|------|-----------|----------|------|
| **SafeTensors** | `.safetensors` | 原始模型权重 | ✅ 支持 |
| **GGUF** | `.gguf` | 量化模型 (Q4_K, Q5_K 等) | ✅ 支持 |
| **ONNX** | `.onnx` | 跨平台推理 | ✅ 支持 |

---

## 2. GGUF 数据结构 (DATA-GGUF)

> **关联需求**: REQ-LOADER-011, REQ-LOADER-014, REQ-LOADER-019
> **关联架构**: ARCH-GGUF-PARSER
> **定位**: gllm 内部使用的 GGUF 解析数据结构

### 2.1 GGUF 值类型 (DATA-GGUF-VALUE-TYPE)

| 类型 ID | 类型名称 | Rust 表示 | 说明 |
|---------|----------|-----------|------|
| 0 | UINT8 | `u8` | 无符号 8 位整数 |
| 1 | INT8 | `i8` | 有符号 8 位整数 |
| 2 | UINT16 | `u16` | 无符号 16 位整数 |
| 3 | INT16 | `i16` | 有符号 16 位整数 |
| 4 | UINT32 | `u32` | 无符号 32 位整数 |
| 5 | INT32 | `i32` | 有符号 32 位整数 |
| 6 | FLOAT32 | `f32` | 32 位浮点数 |
| 7 | BOOL | `bool` | 布尔值 |
| 8 | STRING | `String` | UTF-8 字符串 |
| 9 | ARRAY | `GgufArray` | 数组 (可嵌套) |
| 10 | UINT64 | `u64` | 无符号 64 位整数 |
| 11 | INT64 | `i64` | 有符号 64 位整数 |
| 12 | FLOAT64 | `f64` | 64 位浮点数 |

### 2.3 GGUF 数组 (DATA-GGUF-ARRAY)

| 字段 | 类型 | 说明 |
|------|------|------|
| `item_type` | `GgufValueType` | 元素类型 |
| `items` | `Vec<GgufValue>` | 元素列表 (内部使用) |

**关键**: ARRAY[STRING] 解析必须正确处理每个字符串的长度前缀。

### 2.4 GGUF 量化类型 (DATA-GGUF-DTYPE)

> **关联架构**: ARCH-GGUF-PARSER, gllm-kernels ARCH-QUANT-GENERIC
> **支持策略**: 完整支持所有 28 种量化类型，零拷贝传递给 gllm-kernels

> **🚨 关键约束**: GGUF 解析器使用枚举类型标识符，gllm-kernels 使用泛型架构
>
> - **GGUF 层**: `GgmlDType` 枚举，用于文件格式解析
> - **gllm-kernels 层**: `QuantizedStorage<const N: usize, const BITS: u8>` 泛型 trait
> - **适配层**: 负责类型映射，不违反泛型约束

| 类型 ID | 类型名称 | 块大小 | 块字节数 | 说明 |
|---------|----------|--------|----------|------|
| 0 | F32 | 1 | 4 | 32 位浮点 |
| 1 | F16 | 1 | 2 | 16 位浮点 (可转 F32) |
| 2 | Q4_0 | 32 | 2 + 16 | 4 位量化 |
| 3 | Q4_1 | 32 | 2 + 2 + 16 | 4 位量化 (带 min) |
| 6 | Q5_0 | 32 | 2 + 4 + 16 | 5 位量化 |
| 7 | Q5_1 | 32 | 2 + 2 + 4 + 16 | 5 位量化 (带 min) |
| 8 | Q8_0 | 32 | 2 + 32 | 8 位量化 |
| 9 | Q8_1 | 32 | 4 + 4 + 32 | 8 位量化 (带 min) |
| 10 | Q2_K | 256 | 2 + 2 + QK_K/16 + QK_K/4 | 2 位 K 量化 |
| 11 | Q3_K | 256 | 2 + QK_K/4 + QK_K/8 + 12 | 3 位 K 量化 |
| 12 | Q4_K | 256 | 2 + 2 + QK_K/2 + 12 | 4 位 K 量化 |
| 13 | Q5_K | 256 | 2 + 2 + QK_K/2 + QK_K/8 + 12 | 5 位 K 量化 |
| 14 | Q6_K | 256 | 2 + QK_K/2 + QK_K/4 + QK_K/16 | 6 位 K 量化 |
| 15 | Q8_K | 256 | 4 + QK_K + QK_K/8 | 8 位 K 量化 |
| 16 | IQ2_XXS | 256 | 2 + QK_K/4 | 2 位 I 量化 XXS |
| 17 | IQ2_XS | 256 | 2 + QK_K/4 + QK_K/32 | 2 位 I 量化 XS |
| 18 | IQ3_XXS | 256 | 2 + QK_K/4 + QK_K/8 | 3 位 I 量化 XXS |
| 19 | IQ1_S | 256 | 2 + QK_K/8 + QK_K/16 | 1 位 I 量化 S |
| 20 | IQ4_NL | 32 | 2 + 16 | 4 位 I 量化 NL |
| 21 | IQ3_S | 256 | 2 + QK_K/4 + QK_K/8 + QK_K/32 + 4 | 3 位 I 量化 S |
| 22 | IQ2_S | 256 | 2 + QK_K/4 + QK_K/16 | 2 位 I 量化 S |
| 23 | IQ4_XS | 256 | 2 + 2 + QK_K/2 + QK_K/64 | 4 位 I 量化 XS |
| 24 | I8 | 1 | 1 | 8 位整数 |
| 25 | I16 | 1 | 2 | 16 位整数 |
| 26 | I32 | 1 | 4 | 32 位整数 |
| 27 | I64 | 1 | 8 | 64 位整数 |
| 28 | F64 | 1 | 8 | 64 位浮点 (可转 F32) |
| 29 | IQ1_M | 256 | QK_K/8 + QK_K/16 + QK_K/32 | 1 位 I 量化 M |
| 30 | BF16 | 1 | 2 | BFloat16 (可转 F32) |
| 34 | TQ1_0 | 256 | 2 + 4*13 | Ternary 1.0 |
| 35 | TQ2_0 | 256 | 2 + 64 | Ternary 2.0 |
| 39 | MXFP4 | 32 | 1 + 16 | MXFP4 |

**gllm-kernels 集成**：
- 零拷贝传递量化数据给 gllm-kernels
- gllm-kernels 负责所有量化数据的定点内积运算 (极化解码)
- 解析器只负责正确读取文件格式

### 2.5 GGUF 元数据 Key Registry (DATA-GGUF-KEYS)

> **目的**: Ω1 真实性原则 - 一切从元数据读取，禁止硬编码默认值

#### 2.5.1 必需元数据 (Ω1 强制)

| Key | 类型 | 说明 |
|-----|------|------|
| `general.architecture` | STRING | 模型架构 (llama, qwen2, mistral 等) |

#### 2.5.2 架构特定元数据

> **格式**: `{arch}.{key}`，其中 `{arch}` 是 `general.architecture` 的值

| Key 模板 | 类型 | 说明 | 示例 |
|----------|------|------|------|
| `{arch}.vocab_size` | UINT32 | 词汇表大小 | `llama.vocab_size` |
| `{arch}.context_length` | UINT32 | 最大上下文长度 | `llama.context_length` |
| `{arch}.embedding_length` | UINT64 | Hidden size | `llama.embedding_length` |
| `{arch}.block_count` | UINT32 | 层数 | `llama.block_count` |
| `{arch}.attention.head_count` | UINT32 | Attention heads | `llama.attention.head_count` |
| `{arch}.attention.head_count_kv` | UINT32 | KV heads (GQA) | `llama.attention.head_count_kv` |
| `{arch}.rope.freq_base` | FLOAT32 | RoPE theta | `llama.rope.freq_base` |

#### 2.5.3 Tokenizer 元数据

| Key | 类型 | 说明 |
|-----|------|------|
| `tokenizer.ggml.model` | STRING | Tokenizer 类型 (gpt2, llama, spm) |
| `tokenizer.ggml.tokens` | ARRAY[STRING] | Token 字符串列表 (**修复 bug**) |
| `tokenizer.ggml.merges` | ARRAY[STRING] | BPE merge 规则 |
| `tokenizer.ggml.scores` | ARRAY[FLOAT32] | Token 分数 |
| `tokenizer.ggml.bos_token_id` | UINT32 | BOS token ID |
| `tokenizer.ggml.eos_token_id` | UINT32 | EOS token ID |

### 2.6 GGUF Tensor Info (DATA-GGUF-TENSOR)

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | String | 张量名称 (如 `token_embd.weight`) |
| `n_dims` | u32 | 维度数量 |
| `dims` | [u64; n_dims] | 形状 |
| `dtype` | u32 | 数据类型 (GgmlDType) |
| `offset` | u64 | 在文件中的偏移量 |

### 2.7 QuantizedTensor 量化张量存储 (DATA-QUANT-TENSOR)

> **关联需求**: REQ-QUANT-001 ~ REQ-QUANT-007
> **实现位置**: `src/loader/mod.rs::QuantizedTensor`

量化 tensor 不经过 `Backend::upload_weights()` 上传，而是以原始 block bytes 存储在 `WeightsHandle.quantized` HashMap 中，在推理时直接传递给 TurboQuant 定点矩阵乘法 (quantized_matmul) 内核，禁止独立解包。

```rust
pub struct QuantizedTensor {
    pub data: Vec<u8>,           // 原始 GGUF block bytes
    pub quant_type: QuantType,   // gllm-kernels QuantType 枚举
    pub shape: Vec<usize>,       // 逻辑形状 (行×列)
    pub ggml_dtype: GgmlDType,   // 原始 GGUF 量化类型
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `data` | `Vec<u8>` | 量化 block 原始字节，零转换保留 |
| `quant_type` | `QuantType` | gllm-kernels 量化类型，用于 kernel 分发 |
| `shape` | `Vec<usize>` | 张量逻辑形状 |
| `ggml_dtype` | `GgmlDType` | GGUF 文件中的原始量化类型标识 |

**类型映射 (GgmlDType → QuantType)**:

| 族 | GgmlDType | QuantType | 分发目标 |
|----|-----------|-----------|----------|
| K-Quant | Q2_K ~ Q8_K | Q2K ~ Q8K | `kquant_matmul` |
| Classic | Q4_0 ~ Q8_1 | Q4_0 ~ Q8_1 | `classic_matmul` |
| IQ | IQ1_S ~ IQ4_XS | IQ1S ~ IQ4XS | `iq_matmul` |
| Native | F32/F16/BF16/F64/I* | (None) | `upload_weights` 常规路径 (直接加载，JIT 根据 QuantType 生成对应内核) |

---

| 架构 | `general.architecture` 值 | 特殊 Keys |
|------|--------------------------|-----------|
| Llama | `llama` | `llama.rope.scaling.type`, `llama.rope.scaling.factor` |
| Qwen2/Qwen2.5 | `qwen2`, `qwen2_5` | `qwen2.vocab_size`, `qwen2.rope.alpha` |
| Qwen3 | `qwen3` | 待定义 |
| Mistral | `mistral` | `mistral.sliding_window` |
| Gemma/Gemma2 | `gemma`, `gemma2` | `gemma.attention_logit_softcapping` |
| DeepSeek | `deepseek` | `deepseek.moe` 相关 keys |
| GPT-2/GPT-OSS | `gpt2`, `gpt_neox` | `c_attn`, `c_proj` 融合权重 |

---

## 3. ONNX 数据结构 (DATA-ONNX)

> **关联需求**: REQ-LOADER-012, REQ-LOADER-015

**说明**: ONNX 解析器设计类似 GGUF，遵循 Ω1 真实性原则。

---

## 4. SafeTensors 数据结构 (DATA-SAFE)

> **关联需求**: REQ-LOADER-004

### 4.1 SafeTensors 文件结构

| 组件 | 说明 |
|------|------|
| Header (JSON) | 8 字节长度 + JSON 元数据 |
| Metadata | tensor 名称 → {dtype, shape, offsets} |
| Data | 扁平化的张量数据 |

### 4.2 SafeTensors 数据类型

| 类型代码 | Rust 类型 | 字节数 |
|---------|-----------|--------|
| F32 | `f32` | 4 |
| F16 | `f16` | 2 |
| BF16 | `bf16` | 2 |
| I32 | `i32` | 4 |
| I64 | `i64` | 8 |
| U8 | `u8` | 1 |
| U16 | `u16` | 2 |
| U32 | `u32` | 4 |
| U64 | `u64` | 8 |
| BOOL | `u8` (bit-packed) | 1/8 |

---

## 5. 模型配置数据结构 (DATA-MODEL-CONFIG)

> **关联需求**: REQ-LOADER-007, REQ-LOADER-021

### 5.1 通用配置字段 (Static TurboQuant Bounds)

在 Mega-Kernel 架构下，所有加载的权重由 QuantType 直接驱动 JIT 生成对应的硬件原生内核。gllm 在推理过程中执行 TurboQuant 运行时数学优化（在线 FWHT 旋转 + KV 非对称量化 + RaBitQ 无偏修正），使推理精度逼近数学无损（详见 02-ARCHITECTURE.md §11）。
因此，抛弃旧有动态分配显存的 `dtype_size` 计算机制，统一以物理引擎预编译 DType 约束！

| 字段 | 类型 | 说明 |
|------|------|------|
| `hidden_size` | usize | 隐藏层维度 (Embedding size) |
| `num_hidden_layers` | usize | 模型层数 |
| `num_attention_heads` | usize | 注意力头数 (Q heads) |
| `num_key_value_heads` | usize | 键值对头数 (KV heads / GQA) |
| `intermediate_size` | Option<usize> | FFN 中间层维度 (Expansion size) |
| `num_experts` | Option<usize> | MoE 专家总数 |
| `expert_intermediate_size` | Option<usize> | MoE 专家 FFN 中间层维度 |
| `vocab_size` | usize | 词汇表大小 |
| `max_position_embeddings` | usize | 最大序列长度 (Context window) |
| `head_dim` | usize | 单头维度 (通常为 hidden_size / num_heads) |
| `rope_theta` | f32 | RoPE 基频 |

### 5.2 Ω1 真实性原则加载 

```rust
/// 配置从模型文件读取（禁止硬编码默认值）
pub struct ModelConfig {
    pub vocab_size: u64,
    pub hidden_size: u64,
    pub num_layers: u64,
    pub num_heads: u64,
    pub num_kv_heads: u64,
    pub max_context: u64,
    pub rope_theta: f32,
}

impl ModelConfig {
    /// 从 GGUF 元数据加载
    pub fn from_gguf(reader: &GgufReader) -> Result<Self, GgufError> {
        let arch = reader.architecture()?;

        Ok(Self {
            // Ω1: 从元数据读取，禁止默认值
            vocab_size: reader.get_metadata_u64(&format!("{}.vocab_size", arch))
                .ok_or_else(|| GgufError::MissingMetadata("vocab_size"))?,
            hidden_size: reader.get_metadata_u64(&format!("{}.embedding_length", arch))
                .ok_or_else(|| GgufError::MissingMetadata("embedding_length"))?,
            // ... 其他字段类似
        })
    }
}
```

---

## 6. 张量驱动配置推导 (DATA-TENSOR-DRIVEN)

> **核心原则**: ARCH-TENSOR-DRIVEN
> **关联需求**: REQ-LOADER-022, REQ-LOADER-023

### 6.1 配置优先级

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 1 (最高) | 张量形状 | 从实际权重张量推导配置 |
| 2 | 模型元数据 | GGUF metadata / ONNX attributes / gllm.config |
| 3 (最低) | config.json | 仅作为最后回退 |

### 6.2 通用推导规则

| 配置字段 | 推导方法 | 张量来源 |
|----------|----------|----------|
| `vocab_size` | 从 embedding tensor shape 取较大维度 | `token_embd`, `embed_tokens`, `wte`, `word_embeddings` |
| `head_dim` | 从 Q projection tensor: `q_out / num_heads` | `attn_q`, `q_proj`, `self_attn.q_proj`, `attention.wq` |
| `hidden_size` | 从 embedding tensor 或 attention tensor | 同 `vocab_size` 来源 |
| `intermediate_size`| 从 MLP 投影张量输出维度推导 | `mlp.gate_proj.weight`, `mlp.up_proj.weight`, `blk.{N}.ffn_gate.weight` |

### 6.3 格式特定推导规则

#### 6.3.1 GGUF 格式 (已实现)

| 字段 | 推导方法 | 回退策略 |
|------|----------|----------|
| `vocab_size` | `token_embd.weight` shape 较大维度 | `{arch}.vocab_size` 元数据 |
| `head_dim` | `attn_q.weight` shape / `num_heads` | `attention.head_dim` 或 `rope.dimension_count` 元数据 |

**实现位置**: `src/model_config.rs::from_gguf_loader()`

#### 6.3.2 SafeTensors 格式 (已实现 - REQ-LOADER-022)

| 字段 | 推导方法 | 回退策略 |
|------|----------|----------|
| `vocab_size` | `model.embed_tokens.weight` 等 shape 较大维度 | `gllm.config` 元数据 → `config.json` |
| `head_dim` | `q_proj.weight` shape / `num_heads` | `gllm.config` 元数据 → `config.json` |
| `intermediate_size`| `mlp.gate_proj.weight` / `mlp.up_proj.weight` 输出维度 | `gllm.config` 元数据 → `config.json` |

**张量名称模式**:
- Embedding: `model.embed_tokens.weight`, `embeddings.word_embeddings.weight`, `transformer.wte.weight`
- Q Projection: `model.layers.{N}.self_attn.q_proj.weight`, `encoder.layer.{N}.attention.self.query.weight`
- MLP/FFN: `model.layers.{N}.mlp.gate_proj.weight`, `model.layers.{N}.mlp.up_proj.weight`

**实现要求**:
1. 新增 `from_safetensors_loader_tensor_driven()` 方法
2. 遍历张量列表查找 embedding/Q projection
3. 优先使用张量形状，仅在找不到时回退

#### 6.3.3 ONNX 格式 (已实现 - REQ-LOADER-023)

| 字段 | 推导方法 | 回退策略 |
|------|----------|----------|
| `vocab_size` | `embed_tokens.weight` 等 initializer shape 较大维度 | ONNX graph attributes → 外部 config.json |
| `head_dim` | `q_proj.weight` / `query.weight` initializer shape / `num_heads` | ONNX graph attributes → 外部 config.json |
| `intermediate_size`| `mlp.gate_proj.weight` 等 initializer 输出维度 | ONNX graph attributes → 外部 config.json |

**Initializer 名称模式**:
- Embedding: `embed_tokens.weight`, `embeddings.word_embeddings.weight`, `wte.weight`
- Q Projection: `layers.{N}.self_attn.q_proj.weight`, `encoder.layer.{N}.attention.self.query.weight`
- MLP/FFN: `layers.{N}.mlp.gate_proj.weight`, `layers.{N}.mlp.up_proj.weight`

**实现要求**:
1. 新增 `from_onnx_loader()` 方法
2. 遍历 graph.initializer 查找 embedding/Q projection
3. 优先使用 initializer 形状，仅在找不到时回退

### 6.4 Ω1 合规性

- **严禁默认值**: 如果元数据、张量形状和 `config.json` 都无法提供必需字段，必须返回 `InvalidConfig` 错误，禁止使用硬编码猜测值（如假设 `head_dim=128`）。
- **张量优先**: 即使元数据存在，也应优先尝试从张量形状验证或推导，以确保推理执行时的内存布局绝对正确。

```rust
// ✅ 正确: 从张量推导
let vocab_size = embedding_tensor.shape.iter().max();
let head_dim = q_tensor.shape.max() / num_heads;

// ❌ 错误: 信任可能不准确的元数据
let vocab_size = metadata.get("vocab_size")?;
let head_dim = hidden_size / num_heads;  // 假设标准比例

// ❌ 错误: 使用硬编码默认值
let head_dim = config.head_dim.unwrap_or(128);
let vocab_size = config.vocab_size.unwrap_or(32000);
```

### 6.5 实现状态

| 格式 | 状态 | 关联需求 |
|------|------|----------|
| GGUF | ✅ 已实现 | - |
| SafeTensors | ✅ 已实现 | REQ-LOADER-022 |
| ONNX | ✅ 已实现 | REQ-LOADER-023 |

### 6.6 静态执行约定 (DATA-STATIC-EXECUTION)

> **关联架构**: unified-jit-architecture-master.md §2, 02-ARCHITECTURE.md §11 (TurboQuant)
> **铁律**: gllm 支持加载 SafeTensors/GGUF/ONNX 全格式权重文件。JIT 根据加载时检测到的 QuantType 生成专用内核，推理过程中无类型判断分支。

#### 精度传播路径

所有的执行与编译节点由 QuantType 驱动锁定：

| 张量角色 | 计算通道 | 内存通道 | 传播约束 |
|-------------|-------|-------------|---------|
| GEMM weight | JIT 烧录静态寻址 | KV-Frozen 页表侧载 | 在 GPU Launch 前即通过 Kernel 模板宏固化 |
| Norm weight | JIT 融合缩放节点 | 同上 | 吸收进最终计算中 |
| Activation / intermediate | 静态开辟栈顶 | 伴随 Kernel 随开随毁 | **禁止**返回给 Host 端！ |
| KV cache | Paged Block (Dual-Track) | 根据 QuantType 分配 | `gpu_alloc_kv_paged_pool(quant_type)` |

#### OOM 极高危行为禁止 (OOM Fallback 禁止)

由于 JIT 已根据 QuantType 生成专用内核，排除了运行时 `Amax` 动态探测。如果仍然触发内存或尺寸不匹配错误，引擎不再进行 `GPU → CPU` 或者 `FP16 → FP32` 的动态降级保护！
如果物理内存超过了 Dual-Track 的块总容量，或者编译参数和静态张量形状不匹配，必须**当场崩溃 (Halt)** 并报硬件越界错误。

## 7. 通用张量拓扑 (DATA-TENSOR-TOPOLOGY)

> **关联需求**: REQ-LOADER-022, REQ-LOADER-023
> **关联架构**: ARCH-LOADER-UNIVERSAL

### 7.1 张量角色 (TensorRole)

用于标识张量在 Transformer 架构中的语义作用，独立于具体张量名称。

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    Embedding,          // token_embd, wte
    AttentionQuery,     // q_proj, wq
    AttentionKey,       // k_proj, wk
    AttentionValue,     // v_proj, wv
    AttentionOutput,    // o_proj, wo
    LayerNorm,          // attn_norm, ffn_norm
    FfnGate,            // gate_proj, w1
    FfnDown,            // down_proj, w2
    FfnUp,              // up_proj, w3
    OutputHead,         // lm_head, output
    Unknown,
}
```

### 7.2 模型拓扑 (ModelTopology)

描述模型权重的逻辑结构，用于指导加载和配置推导。

```rust
pub struct ModelTopology {
    /// 张量映射表: Role -> (LayerIdx -> TensorName)
    pub tensors: HashMap<TensorRole, Vec<String>>,
    /// 识别出的最大层数
    pub num_layers: usize,
}
```

## 8. 调度器重构数据结构 (DATA-SCHED-REFACTOR)

> **关联需求**: REQ-SCHED-015, REQ-SCHED-016, REQ-SCHED-017, REQ-SCHED-018, REQ-KV-001, REQ-KV-002, REQ-KV-003, REQ-KV-004
> **关联架构**: ARCH-SCHED-REFACTOR-2026

### 8.1 KvPrefixIndex 前缀树 (DATA-KV-PREFIX-INDEX)

用于在无 Session ID 场景下查找最长可复用 token 前缀。

```rust
pub struct KvPrefixIndex {
    pub root: TrieNode,
}

pub struct TrieNode {
    pub children: HashMap<TokenId, TrieNode>,
    pub page_ref: Option<PageRef>,
}

pub struct PageRef {
    pub virtual_page_id: VirtualPageId,
    pub offset_in_page: usize,
}

pub struct PrefixMatch {
    pub matched_tokens: usize,
    pub matched_pages: Vec<VirtualPageId>,
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `children` | `HashMap<TokenId, TrieNode>` | 按 token 扩展的前缀边 |
| `page_ref.virtual_page_id` | `VirtualPageId` | 命中 token 对应的 KV 页面 |
| `page_ref.offset_in_page` | `usize` | token 在页面中的偏移 |
| `matched_tokens` | `usize` | 可复用前缀长度 |

### 8.2 SessionKvCache 会话缓存 (DATA-KV-SESSION-CACHE)

用于会话级确定性 append 复用。

```rust
pub struct SessionKvCache {
    pub session_id: SessionId,
    pub pages: Vec<VirtualPageId>,
    pub finalized_position: usize,
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | `SessionId` | 会话唯一标识 |
| `pages` | `Vec<VirtualPageId>` | 已确认前缀对应的页面序列 |
| `finalized_position` | `usize` | 已确认 token 边界（单调递增） |

### 8.3 KvPipeline 双管线标识 (DATA-KV-PIPELINE)

用于分离可复用会话内容与可丢弃工作内容。

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvPipeline {
    Conversation,
    Working,
}

pub struct PipelinedVirtualPageId {
    pub pipeline: KvPipeline,
    pub sequence_id: RequestId,
    pub logical_index: usize,
}
```

| 枚举值 | 语义 | 生命周期 |
|--------|------|----------|
| `Conversation` | System/User/Assistant 主对话上下文 | 跨轮保留 |
| `Working` | Thinking/Reasoning 临时上下文 | 轮次结束可回收 |

### 8.4 批顺序策略 (DATA-BATCH-ORDER-POLICY)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchOrderPolicy {
    StrictRequestIdOrder,
    FifoOrder,
}
```

| 策略 | 排序键 | 约束 |
|------|--------|------|
| `StrictRequestIdOrder` | `RequestId` | 默认策略，确定性优先 |
| `FifoOrder` | `enqueue_time` | 确定性 FIFO |

### 8.5 Prefill/Decode 页面规划 (DATA-PREFILL-PLAN)

```rust
pub enum PrefillPlan {
    FullyResident {
        pages: usize,
    },
    Pipelined {
        l1_pages: usize,
        l2_prefetch: usize,
        chunk_schedule: Vec<usize>,
    },
}
```

用于表达 `ChunkedConfig` 融合后的预分配与预取策略，约束如下：
1. `FullyResident` 表示 prefill 页面可全部驻留 L1。
2. `Pipelined` 表示需要分批装载/预取，不改变 Phase Isolation。
3. `chunk_schedule` 仅在 Prefill 阶段生效，不参与 Decode 混批。

---

## 9. 硬件能力与 Codegen 维度 (DATA-HW-CODEGEN)

> **关联需求**: REQ-ARCH-001, REQ-ARCH-003
> **关联架构**: 02-ARCHITECTURE.md §5 (JIT 编译管线), ARCH-DETAILED-DESIGNS.md 功能 7
> **核心原则**: JIT codegen 的最终机器码由 `f(算法语义, ISA, DType, 设备参数)` 的笛卡尔积决定

### 9.1 Codegen 维度模型 (DATA-CODEGEN-DIMENSIONS)

JIT 编译管线的 Phase 3 (ISA Lowering) 根据以下 6 个正交维度生成最优机器码：

| 维度 | 类型 | 影响范围 | 示例 |
|------|------|---------|------|
| **ISA 级别** | `IsaLevel` | 指令集选择、向量宽度 | AVX2 vs AVX-512 vs NEON vs SVE |
| **ISA 扩展** | `IsaFeatures` | 特化指令路径 | VNNI, BF16, FP16, AMX, SVE2 |
| **量化类型** | `QuantType` | 权重量化格式特化 | Q4_K / Q8_K / F16 / ... |
| **寄存器预算** | `TargetDesc` | Tile 大小、溢出策略、SW pipeline 深度 | 16 ymm vs 32 zmm vs 32 v-reg |
| **设备参数** | `KernelConfig` | Cache blocking、prefetch、NUMA | L1/L2/L3 大小, cacheline, core count |
| **GPU 计算能力** | `GpuDeviceProfile` | SM 版本特化、shared memory、tensor core 代 | sm_70 vs sm_90 vs gfx1100 |

**铁律**: 这 6 个维度的任何组合都必须通过 JIT 管线动态生成代码，禁止硬编码特定组合。

### 9.2 ISA 级别 (DATA-ISA-LEVEL)

> **实现位置**: `gllm-kernels/src/dispatch/device_profile.rs`

```rust
pub enum IsaLevel {
    Scalar,      // 无 SIMD，仅用于 Phase 0 参考实现
    Avx2,        // x86_64: 256-bit SIMD, 16 ymm regs
    Avx512,      // x86_64: 512-bit SIMD, 32 zmm regs
    Avx512Amx,   // x86_64: AVX-512 + Intel AMX (Sapphire Rapids+)
    Neon,        // AArch64: 128-bit SIMD, 32 v-regs
    Sve,         // AArch64: 可变长 SIMD (128~2048-bit), 32 z-regs
    Sve2,        // AArch64: SVE2 超集
    NeonAmx,     // AArch64: Apple AMX (M1+)
}
```

| 变体 | 向量宽度 (bits) | SIMD 寄存器数 | 典型平台 |
|------|----------------|--------------|---------|
| `Scalar` | — | — | Phase 0 SymExec only |
| `Avx2` | 256 | 16 ymm | Haswell+ (2013) |
| `Avx512` | 512 | 32 zmm | Skylake-X+ (2017) |
| `Avx512Amx` | 512 + tile | 32 zmm + 8 tmm | Sapphire Rapids+ (2023) |
| `Neon` | 128 | 32 v-reg | ARMv8+ 全系 |
| `Sve` | 128~2048 | 32 z-reg | Neoverse V1+ (2021) |
| `Sve2` | 128~2048 | 32 z-reg | Neoverse V2+ (2023) |
| `NeonAmx` | 128 + AMX block | 32 v-reg + AMX | Apple M1+ (2020) |

**分派规则**: `DeviceProfile::detect()` 运行时检测 → `IsaLevel` → codegen 选择 x86_64 / AArch64 后端。

### 9.3 ISA 扩展特性 (DATA-ISA-FEATURES)

> **实现位置**: `gllm-kernels/src/autotuning/hw_info.rs`

```rust
pub struct IsaFeatures {
    // x86_64 扩展
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,     // INT8 dot product (Cascade Lake+)
    // AArch64 扩展
    pub neon: bool,
    pub sve: bool,
    pub sve2: bool,
    pub sve_vl_bytes: usize,  // SVE 向量长度 (0 = SVE 不可用)
}
```

**ISA 扩展 → Codegen 路径映射**:

| 扩展 | 守卫条件 | 激活的 Codegen 路径 |
|------|---------|-------------------|
| `avx512vnni` | `has_vnni` | VPDPBUSD W8A8 (TurboQuant) 张量内联 |
| `sve` | `use_sve && sve_vl_bytes > 0` | SVE 可变长向量 codegen (替代 NEON 固定 128-bit) |
| `amx` (x86) | `has_amx && amx_gemm_eligible(m,n,k)` | Intel AMX tile GEMM (`tdpbssd` 定点规格) |
| `amx` (Apple) | `IsaLevel::NeonAmx` | Apple AMX block 运算 |

**铁律**: 扩展不可用时必须降级到基础 ISA 路径 (AVX2/NEON)，禁止返回错误。这是 SPEC 授权的唯一 ISA 级别 fallback。

### 9.4 量化类型驱动维度 (DATA-QUANTTYPE-DIMENSION)

> QuantType 直接驱动 JIT 内核生成。加载时检测到的 QuantType 决定 JIT 生成哪种硬件原生指令，推理过程中无类型判断分支。

QuantType 约束 Codegen：

| 层面 | 量化类型 (如 Q4_K) | 原生浮点 (如 F16) |
|------|------|------|
| **内存寻址** | 纯 `byte_width` 游标 | 纯 `byte_width` 游标 |
| **向量加速** | 硬件整型内积 (VNNI/WGMMA) | 硬件浮点内积 |
| **存储转换** | 压扁至物理内存页 | 压扁至物理内存页 |

**累加器约束**: 所有内核直接在内部执行微字节累加，仅在系统边界处交付计算产物。

### 9.5 寄存器预算与分配策略 (DATA-REGISTER-BUDGET)

> **实现位置**: `gllm-kernels/src/compiler/codegen/target_desc.rs`

```rust
pub struct TargetDesc {
    pub simd_width_polar: usize, // Polar常量向量宽度 (INT/定点): 对齐物理总线
    pub num_simd_regs: usize,    // 总 SIMD 寄存器数: 16=AVX2, 32=NEON/AVX-512
    pub n_scratch_regs: usize,   // 可用 scratch = total - accumulators - reserved
    pub mr: usize,               // Tile M 维度
    pub nr: usize,               // Tile N 维度
    pub simd_bytes: usize,       // 向量字节数: 16=NEON, 32=AVX2, 64=AVX-512
}
```

**寄存器预算公式**:

```
accumulators = mr × nr_vecs           (nr_vecs = nr / simd_width_polar)
scratch      = n_scratch_regs         (A broadcast + B conversion)
total_used   = accumulators + scratch
free_regs    = num_simd_regs - total_used
```

**寄存器预算约束表**:

| ISA | 总寄存器 | 典型 Tile (mr×nr) | 累加器 | Scratch | 剩余 (epilogue) |
|-----|---------|-------------------|--------|---------|----------------|
| AVX2 | 16 ymm | 6×16 (6×2 vecs) | 12 | 2 | 2 |
| AVX-512 | 32 zmm | 6×48 (6×3 vecs) | 18 | 2 | 12 |
| NEON | 32 v-reg | 8×16 (8×4 vecs) | 32 | 0 | 0 (需溢出) |
| SVE (256-bit) | 32 z-reg | 6×16 (6×2 vecs) | 12 | 2 | 18 |

**寄存器分配作为 Codegen 维度**:

寄存器预算直接决定以下 codegen 行为：

| 寄存器压力 | Tile 大小 | SW Pipeline | Epilogue 策略 |
|-----------|----------|-------------|--------------|
| 充裕 (free ≥ 8) | 最大 tile | depth=2 双缓冲 | 全寄存器 epilogue |
| 适中 (free 4~7) | 标准 tile | depth=1 单缓冲 | 部分溢出 |
| 紧张 (free < 4) | 缩小 tile | depth=0 无流水线 | 栈溢出 (spill/reload) |

### 9.6 设备参数 (DATA-DEVICE-PARAMS)

> **实现位置**: `gllm-kernels/src/microarch.rs`, `gllm-kernels/src/autotuning/hw_info.rs`

#### 9.6.1 硬件信息 (HwInfo)

```rust
pub struct HwInfo {
    pub vendor: String,
    pub model_name: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub l1d_bytes: usize,        // L1 数据缓存
    pub l2_bytes: usize,         // L2 缓存
    pub l3_bytes: usize,         // L3 缓存
    pub cacheline_bytes: usize,  // 缓存行大小 (通常 64)
    pub isa: IsaFeatures,
}
```

#### 9.6.2 内核配置 (KernelConfig)

```rust
pub struct KernelConfig {
    pub arch: MicroArch,
    // 微内核几何
    pub mr: usize,               // Tile M 维度
    pub nr: usize,               // Tile N 维度
    pub simd_width: usize,       // SIMD 极化向量宽度 (元素数)
    // Cache blocking
    pub kc: usize,               // K 维度分块
    pub mc: usize,               // M 维度分块
    pub nc: usize,               // N 维度分块
    // Cache 大小
    pub l1d: usize,
    pub l2: usize,
    pub l3: usize,
    // Prefetch
    pub pf_distance_b: usize,    // B 矩阵预取距离
    pub pf_distance_a: usize,    // A 矩阵预取距离
    pub pf_hint_gemv: u8,        // GEMV 预取提示 (0=T0, 1=T1, 2=T2, 3=NTA)
    pub pf_rows_gemv: usize,     // GEMV 预取行数
    // 硬件特性 (从 IsaFeatures 提取的 codegen 开关)
    pub use_avx512: bool,
    pub zmm_downclocking: bool,  // Zen4 等 AVX-512 降频平台
    pub has_amx: bool,
    pub has_vnni: bool,
    pub has_sve: bool,
    pub has_sve2: bool,
    pub sve_vl_bytes: usize,
}
```

**KernelConfig → Codegen 贯通路径**: `DeviceProfile::detect()` → `MicroArch::kernel_config()` → `X86CodeGen::new(config)` / `DynasmAArch64CodeGen::new(config)`。所有字段在 codegen 构造时一次性注入，运行时不可变。

#### 9.6.3 NUMA 拓扑 (NumaTopology)

```rust
pub struct NumaTopology {
    pub nodes: Vec<NumaNode>,
}

pub struct NumaNode {
    pub node_id: usize,
    pub l3_bytes: usize,
    pub core_count: usize,
}
```

**NUMA → Codegen**: 多节点时 `nc_for_node_l3()` 缩小 NC 分块以适配单节点 L3。

### 9.7 GPU 计算能力 (DATA-GPU-PROFILE)

> **实现位置**: `gllm-kernels/src/gpu/mod.rs`

#### 9.7.1 GPU 平台 (Platform)

```rust
pub enum Platform {
    X86_64 { avx512: bool, amx: bool },
    Aarch64 { sve: bool, amx: bool },
    #[cfg(feature = "jit-cuda")]
    Cuda { sm_version: u32 },    // SM 版本: 70=Volta, 80=Ampere, 90=Hopper, 100=Blackwell
    #[cfg(feature = "jit-hip")]
    Hip { gfx_arch: u32 },       // GFX 架构: 1100=RDNA3, 942=CDNA3
    #[cfg(feature = "jit-metal")]
    Metal { gpu_family: u32 },   // GPU Family: 7=M1, 8=M2, 9=M3
}
```

#### 9.7.2 GPU 设备描述 (GpuDeviceProfile)

```rust
pub struct GpuDeviceProfile {
    pub platform: Platform,
    pub compute_units: u32,
    pub shared_mem_per_block: u32,
    pub max_registers_per_thread: u32,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub max_block_dim: [u32; 3],
    pub max_grid_dim: [u32; 3],
    pub total_memory: usize,
    pub memory_bandwidth_gbs: f64,
    pub peak_tops_polar: f64,     // 替换原有的浮点算力统计，使用极化精度算力 (TOPS)
    pub has_matrix_unit: bool,    // Tensor Core / Matrix Core
    pub clock_mhz: u32,
    pub isv: GpuIsvCapabilities,
}
```

#### 9.7.3 SM 版本分派 (SmRange + PtxKernelRegistry)

```rust
pub struct SmRange {
    pub min_sm: u32,   // 包含
    pub max_sm: u32,   // 不包含 (u32::MAX = 无上界)
}

pub struct PtxKernelRegistry {
    entries: HashMap<PtxAlgorithm, Vec<PtxKernelEntry>>,
}

pub enum PtxAlgorithm {
    FlashAttention,
    Gemm,
}
```

**SM 版本 → Kernel 映射**:

| 算法 | SM 范围 | 关键指令 | Tensor Core 代 |
|------|---------|---------|---------------|
| FlashAttention v1 | [70, 80) | `wmma.load` / `wmma.mma` | Gen 1 (Volta) |
| FlashAttention v2 | [80, 90) | `mma.sync` / `cp.async` | Gen 2 (Ampere) |
| FlashAttention v3 | [90, 100) | `wgmma` / `cp.async.bulk` / TMA | Gen 3 (Hopper) |
| FlashAttention v4 | [100, ∞) | `tcgen05.mma` / TMEM / 2-CTA | Gen 4 (Blackwell) |
| GEMM TC sm70 | [70, 80) | `wmma` | Gen 1 |
| GEMM TC sm80 | [80, 90) | `mma.sync` | Gen 2 |
| GEMM TC sm89+ | [90, ∞) | `mma.sync` + FP8 | Gen 3+ |

**铁律**: SM < 70 必须返回 `Err`，禁止 fallback (REQ-KERNELS-PTX-MV-003)。

#### 9.7.4 GPU 硬件能力 (GpuIsvCapabilities)

```rust
pub struct GpuIsvCapabilities {
    pub tensor_core_gen: u8,  // 0=none, 1=Volta/gfx900, 2=Ampere/CDNA2, 3=Hopper/CDNA3
}
```

> **设计决策**: GPU GEMM 全部走 JIT codegen 生成设备原生二进制（PTX/HIP/MSL），不调用外部 BLAS 库（cuBLAS/rocBLAS）。
> JIT 融合算子直接驻留缓存对齐内存页，调用外部预编译库的 ROI 过低。
> `tensor_core_gen` 作为 JIT codegen 的硬件能力信号，驱动 WMMA/MMA/MFMA 指令选择。

### 9.8 算法策略选择 (DATA-ALGORITHM-STRATEGY)

> **实现位置**: `gllm-kernels/src/compiler/codegen/gemm_dispatch.rs`, `gllm-kernels/src/compiler/graph.rs`

#### 9.8.1 GEMM 策略 (GemmStrategy)

```rust
pub enum GemmStrategy {
    JitBlis,              // JIT 编译 BLIS 微内核 (CPU 默认)
    OneDnn,               // Intel oneDNN ISV 库 (CPU, 大矩阵)
    Amx,                  // Intel AMX tile GEMM (CPU, BF16)
    AppleAmx,             // Apple AMX block GEMM (CPU, macOS)
    JitGpuTensorCore,     // JIT GPU GEMM + Tensor Core/MFMA
    JitGpu,               // JIT GPU GEMM (无矩阵加速单元)
}
```

**选择决策树**: `select_gemm_strategy()` 按优先级：GPU + tensor_core_gen → JitGpuTensorCore / JitGpu → CPU AMX → oneDNN → Accelerate → JIT BLIS。

#### 9.8.2 Attention 策略 (AttentionStrategy)

```rust
pub enum AttentionStrategy {
    Naive,                                          // O(n²) 参考实现
    FlashV2 { block_m: usize, block_n: usize },    // FlashAttention-2 tiled
    Paged { page_size: usize },                     // PagedAttention (vLLM-style)
    SlidingWindow { window_size: usize },           // 滑动窗口 (Mistral-style)
}
```

**选择依据**: `select_attention_strategy()` 根据 L1 cache 阈值感知自动选择。

### 9.9 Codegen 提示 (DATA-CODEGEN-HINTS)

> **实现位置**: `gllm-kernels/src/compiler/semantic_dag.rs`

```rust
pub struct CodegenHints {
    pub is_memory_bound: bool,       // Roofline 分类: 内存密集型
    pub arithmetic_intensity: f32,   // 算术强度 (FLOP/byte)
    pub prefetch_hint: u8,           // 预取策略: 0=T0, 1=T1, 2=T2, 3=NTA
    pub use_nt_stores: bool,         // 非临时存储 (bypass cache)
}
```

**SemanticDAG → CodegenHints → Codegen**: Phase 1 分析算子 AI/bottleneck → 聚合为 hints → Phase 3 codegen 据此调整预取距离、NT store、融合决策。

### 9.10 DeviceProfile 聚合 (DATA-DEVICE-PROFILE)

> **实现位置**: `gllm-kernels/src/dispatch/device_profile.rs`

```rust
pub struct DeviceProfile {
    pub arch: MicroArch,
    pub isa: IsaLevel,
    pub kernel_config: KernelConfig,
    pub hw_info: HwInfo,
    pub numa: NumaTopology,
    pub peak_tops_polar: f64, // 使用极化精度算力 (TOPS) 替换废弃的 peak_gflops_f32
    pub peak_bandwidth_gbs: f64,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub isv: IsvCapabilities,
}
```

**关键方法**:

| 方法 | 返回 | 用途 |
|------|------|------|
| `detect()` | `DeviceProfile` | 运行时自动检测所有硬件能力 |
| `num_simd_regs()` | `usize` | 寄存器预算计算 |
| `simd_width()` | `usize` | 定点极化元素/向量宽度 |
| `cache_sizes()` | `(l1, l2, l3)` | Cache blocking 决策 |
| `gemm_blocking(m, n, k)` | `GemmBlocking` | MC/NC/KC 分块参数 |
| `roofline_ridge_point()` | `f64` | Roofline 模型拐点 (FLOP/byte) |

**DeviceProfile 是所有 codegen 维度的聚合入口**: 从 `detect()` 开始，经过 `KernelConfig` 注入 codegen，驱动 Phase 3 ISA Lowering 的所有分派决策。

### 9.11 物理 DType 维度 (DATA-DTYPE-SPECIALIZATION)

> **实现位置**: `gllm-kernels/src/compiler/codegen/x86_64.rs`, `gllm-kernels/src/compiler/graph.rs`

量化 GEMM 统一被纳入 TurboQuant 加载时 DType 特化体系，针对各精度在模型加载时一次性生成指令级定点融合路径：

```rust
pub enum OpKind {
    // ...
    QuantGemm { m: usize, n: usize, k: usize, bits: u8 },
}
```

| 精度/DType | 定点内联映射 (原 Dequant 路径) | 内存布局 | 指令特化 |
|------|-------------|---------|---------|
| 8-bit (Q8_0) | `emit_dequant_q8_0_loop()` — 逐 block 定点展开×i8 | 32 元素/block, 2B scale + 32B data | VNNI 可用时 `vpdpbusd` |
| 4-bit (Q4_0) | `emit_dequant_q4_0_loop()` — 极化解码 + 定点映射 | 32 元素/block, 2B scale + 16B data | 位操作 `vpand`/`vpsrlw` |
| 其他 | `Err("unsupported bits")` | — | — |

**与 TurboQuant 的关系**: `QuantGemm` 源数据为定点整数，系统根据 QuantType 在后端产生硬件原生整数乘加指令链，禁止执行浮点解包。

### 9.11.1 JIT 算子覆盖矩阵 (DATA-OPKIND-COVERAGE)

> **实现位置**: `gllm-kernels/src/compiler/codegen/x86_64.rs`, `aarch64_dynasm.rs`, `gpu_ir/`

所有算子必须走完整 JIT 管线（Scalar→SymExec→IR→ISA Lowering），以下为三平台覆盖状态：

| OpKind | x86_64 | AArch64 | GPU (PTX/HIP/MSL) | 备注 |
|--------|--------|---------|-------------------|------|
| Gemm | ✅ | ✅ | ✅ | BLIS 微内核 |
| GemmBias | ✅ | ✅ | ✅ | Gemm + EpilogueInjection bias |
| QuantGemm | ✅ | ✅ | ✅ | Dequant 融合进 GEMM（见 §9.11.2） |
| RmsNorm | ✅ | ✅ | ✅ | |
| LayerNorm | ✅ | ✅ | ✅ | |
| RoPE | ✅ | ✅ | ✅ | |
| Softmax | ✅ | ✅ | ✅ | |
| Silu | ✅ | ✅ | ✅ | |
| Gelu | ✅ | ✅ | ✅ | |
| SwiGlu | ✅ | ✅ | — | GPU 走图优化层 SwiGLU 融合 |
| GeGlu | — | ✅ | — | AArch64 独立实现 |
| Add / Mul / Residual | ✅ | ✅ | ✅ | |
| MultiHeadAttention | ✅ | ✅ | ✅ | |
| CachedGQA | ✅ | ✅ | ✅ (trace) | 带 KV cache 的 GQA |
| MoEGate | ✅ | ✅ | ✅ (trace) | |
| TopK | ✅ | ✅ | ✅ (trace) | |
| WeightedSum | ✅ | ✅ | ✅ (trace) | MoE expert 加权合并 |
| MeanPool | ✅ | ✅ | ✅ | Encoder 平均池化 |
| L2Normalize | ✅ | ✅ | ✅ | Embedding 归一化 |

| Reshape / Transpose | NOP | NOP | NOP | 纯元数据操作，不生成指令 |

### 9.11.2 算子融合关系 (DATA-OP-FUSION-MAP)

算子之间存在两层融合：**gllm-kernels Phase 2 硬件融合** 和 **gllm 图优化层模式融合**。

#### Phase 2 硬件融合模式 (gllm-kernels)

> **实现位置**: `gllm-kernels/src/compiler/fusion.rs`, `hw_constraints.rs`

| FusionMode | 融合关系 | 触发条件 |
|------------|---------|---------|
| EpilogueInjection | GEMM + 激活/bias/残差 → 融合到累加器寄存器 | 后继是 Silu/Gelu/Add/Bias |
| LoopFusion | 逐元素算子链 → 合并为单循环 | 连续 Elementwise ops |
| TileLevelFusion | 前驱算子嵌入 GEMM MC 循环 | 前驱输出 > 75% L1 |
| ComputeRoot | 前驱算子完整计算后驻留 L1/L2 | 前驱输出 ≤ 75% L1 |
| QkvSharedInput | Q/K/V 三个 GEMM 共享 pack_a | 连续 3 个 GEMM 共享输入 |
| NormIntoGemm | RmsNorm 输出直接喂入 GEMM | RmsNorm → GEMM 无中间写回 |
| Standalone | 不融合，独立执行 | 硬件约束违反时降级 |

#### 关键融合实例

**TurboQuant × StaticGemm 压缩内联积**（多平台特性）：
```
独立浮点 Dequantize op → ❌ 架构禁止生成
QuantGemm 根据 QuantType 发射硬件整型内置算符（如 `vpdpbusd` / `sdot`）
  → 零中间解包显存驻留，数据在寄存器流水线中直接完成定点内积运算 (VNNI)
  → 仅输出累计标量，无全生命周期的精度膨胀负担
```
全架构（AArch64/x86/GPU）同步剔除原 `Dequantize` 回退 OpKind，严格服从纯整数压扁。

**FFNBlock 深度融合**（三平台）：
```
Gate GEMM + Up GEMM → fused activation × multiply → Down GEMM
  → Gate/Up 共享 pack_a (QkvSharedInput 模式)
  → activation×mul 融合到 Gate GEMM epilogue (EpilogueInjection)
```

**CrossLayerResidual 深度融合**（三平台）：
```
Add (残差) → RmsNorm → GEMM
  → Add 输出直接进 RmsNorm scratchpad (ComputeRoot)
  → RmsNorm 输出直接喂 GEMM (NormIntoGemm)
  → 跨层零中间写回
```

#### gllm 图优化层模式融合

> **实现位置**: `gllm/src/graph/optimizer/`

| Pattern Pass | 融合关系 | 输入子图 → 输出节点 |
|-------------|---------|-------------------|
| FlashAttention | Q×K^T → Scale → Mask → Softmax → ×V | → FlashAttn 单节点 |
| GQA | Multi-head attention + KV repeat | → GQA 单节点 |
| FusedQkvRope | Q/K/V projection + RoPE | → QkvRope 单节点 |
| SwiGLU | Gate GEMM × Silu × Up GEMM | → SwiGLU 单节点 |
| MoERouting | Gate → TopK → Expert FFN → WeightedSum | → MoE 单节点 |
| FusedRMSLinear | RmsNorm → Linear | → RMSLinear 单节点 |
| HardwareFusionPass | 硬件不支持时降级到 Atomic | 按 DeviceProfile 决策 |
| ConstantFolding | 常量表达式预计算 | 编译期求值 |
| DeadCodeElimination | 移除未使用节点 | 图瘦身 |

### 9.12 JitParams 自动调优维度 (DATA-JITPARAMS)

> **实现位置**: `gllm-kernels/src/autotuning/search_space.rs`

JitParams 是运行时自动调优的参数空间，每个参数都是独立的 codegen 分派维度：

```rust
pub struct JitParams {
    pub k_unroll: usize,
    pub prefetch_distance: usize,
    pub reg_alloc_strategy: RegAllocStrategy,
    pub sw_pipeline_depth: usize,
    pub nr_variant: usize,
}

pub enum RegAllocStrategy {
    MaxAccumulators,  // 2 scratch regs — 最大化累加器 tile
    Balanced,         // 4 scratch regs — 平衡
    MinSpill,         // 6 scratch regs — 最小化寄存器溢出
}
```

| 参数 | 搜索范围 | 影响的 Codegen 行为 |
|------|---------|-------------------|
| `k_unroll` | 1, 2, 4, 8 | K 循环展开因子，影响指令级并行度 |
| `prefetch_distance` | 0~16 | B 矩阵预取距离 (0=禁用)，影响 cache miss 率 |
| `reg_alloc_strategy` | 3 种 | Scratch 寄存器数量，影响 tile 大小上限 |
| `sw_pipeline_depth` | 0, 1, 2 | 软件流水线深度 (0=无, 1=单缓冲, 2=双缓冲) |
| `nr_variant` | 0 或具体值 | NR tile 宽度覆盖 (0=使用 blocking 默认值) |

**与静态维度的关系**: §9.1~§9.7 的维度在 `DeviceProfile::detect()` 时确定，运行期间不变。JitParams 是在静态维度确定后，通过 autotuning 搜索在参数空间中找到最优组合。两层维度的笛卡尔积构成完整的 codegen 空间。

### 9.13 硬件加速 Eligibility 阈值 (DATA-HW-ELIGIBILITY)

> **实现位置**: 各 codegen 后端

硬件加速单元（AMX、Tensor Core、Matrix Core）有最小问题规模要求，低于阈值时降级到通用路径：

#### 9.13.1 Intel AMX (x86_64)

```rust
fn amx_gemm_eligible(m: usize, n: usize, k: usize, tdpbssd_ready: bool) -> bool {
    tdpbssd_ready && m >= 32 && n >= 32 && m % 16 == 0 && n % 16 == 0 && k > 0
}
```

- 不满足 → 降级到 JIT BLIS 路径

#### 9.13.2 Apple AMX (AArch64)

```rust
fn apple_amx_gemm_eligible(m: usize, n: usize, k: usize) -> bool {
    m >= 32 && n >= 32 && m % 32 == 0 && n % 32 == 0 && k > 0
}
```

- 不满足 → 降级到 NEON BLIS 路径

#### 9.13.3 CUDA Tensor Core

| SM 范围 | 最低要求 | 降级路径 |
|---------|---------|---------|
| sm < 70 | — | `Err` (禁止 fallback) |
| sm ≥ 70 | `has_matrix_unit = true` | 标量 tiled GEMM |

#### 9.13.4 HIP Matrix Core (MFMA)

| GFX 架构 | 条件 | 降级路径 |
|----------|------|---------|
| gfx < 908 | 无 MFMA | 标量 tiled GEMM |
| gfx ≥ 908 | `v_mfma_f32_16x16x16f16` | — |

**Warp/Wave 大小**: `gfx ≥ 1000` (RDNA) → wave32; `gfx < 1000` (CDNA) → wave64

#### 9.13.5 Metal Simdgroup Matrix

| GPU Family | 条件 | 降级路径 |
|-----------|------|---------|
| < 7 | 无 simdgroup matrix | 标量 tiled GEMM |
| ≥ 7 (M1+) | `simdgroup_multiply_accumulate` | — |

### 9.14 完整维度分类总览 (DATA-DIMENSION-TAXONOMY)

```
┌─────────────────────────────────────────────────────────────────┐
│  Codegen 维度分类                                                │
│                                                                  │
│  第一层: 编译时维度 (Cargo feature)                               │
│  ├── GPU 后端选择: jit-cuda / jit-hip / jit-metal               │
│                                                                  │
│  第二层: 检测时维度 (DeviceProfile::detect(), 运行期不变)         │
│  ├── ISA 级别: IsaLevel (AVX2/AVX-512/NEON/SVE/...)            │
│  ├── ISA 扩展: IsaFeatures (VNNI/BF16/FP16/AMX/SVE2/...)      │
│  ├── 微架构: MicroArch (tile 几何/prefetch/FMA ports)           │
│  ├── 设备参数: HwInfo (cache/cacheline/cores)                   │
│  ├── NUMA 拓扑: NumaTopology (节点数/per-node L3)               │
│  ├── GPU 能力: GpuDeviceProfile (SM/GFX/Family/shared_mem)     │
│  └── ISV 库: IsvCapabilities (CPU: oneDNN/Accelerate)           │
│                                                                  │
│  第三层: 算子时维度 (per-op, 每个算子可能不同)                    │
│  ├── 量化类型: QuantType                                          │
│  ├── 物理精度: bits (4/8)                                       │
│  ├── 算法策略: GemmStrategy / AttentionStrategy                 │
│  ├── Codegen 提示: CodegenHints (memory_bound/AI/prefetch)      │
│  └── HW Eligibility: 加速单元最小规模阈值                        │
│                                                                  │
│  第四层: 调优时维度 (JitParams, autotuning 搜索)                 │
│  ├── k_unroll: K 循环展开因子                                    │
│  ├── prefetch_distance: 预取距离                                 │
│  ├── reg_alloc_strategy: 寄存器分配策略                          │
│  ├── sw_pipeline_depth: 软件流水线深度                           │
│  └── nr_variant: NR tile 宽度覆盖                               │
│                                                                  │
│  最终机器码 = f(第一层 × 第二层 × 第三层 × 第四层)               │
└─────────────────────────────────────────────────────────────────┘
```

**维度总数**: 27 个独立分派点，分布在 4 个层级。JIT 编译的核心价值在于：预编译库无法覆盖这 27 个维度的笛卡尔积，而 JIT 管线在运行时根据实际组合动态生成最优代码。

### 9.6 `HardwareIR` 物理能力抽象 (DATA-HARDWARE-IR)

> **关联架构**: 02-ARCHITECTURE.md §12.5 (Silicon-Level Instruction Mapping)
> **设计思想**: 除了描述逻辑形状的 `SymDim` 之外，编译器在生成硅晶映射所需的底层汇编时，必须依赖明确的硬件能力 IR。

```rust
/// JIT 编译器在执行 Code-Generation 时强制读取的硬件物理探测结构
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HardwareIR {
    /// Intel AMX: 指定最多可用的矩阵寄存器横纵宽限
    AmxTile { max_rows: usize, max_cols: usize },
    
    /// AVX10 混合收敛向量: 强制将物理切片降解匹配 E-Core 的 256位
    Avx10Converged { vector_width: usize }, 
    
    /// Hopper TMA: 张量内存加速器特化的描述，直接生成异步 LDG 块
    Sm90Tma { block_width: usize, block_height: usize },
    
    /// Blackwell FP4/FP6 Native Tension
    Sm100NativeMma { quant_type: QuantType },
    
    /// AMD CDNA3 的 XCD 屏障边界
    Cdna3XcdBarrier { max_workgroup_size: usize },
}
```

### 9.7 编译器约束变量数据结构 (DATA-COMPILER-CONSTRAINTS)

> **关联架构**: 02-ARCHITECTURE.md §12.6 (MicroArch-to-IR Constraints)
> **设计思想**: 将物理传感器探测结果坍缩为 JIT 编译器的强类型约束。

```rust
/// 探测器会在 Load-Time 阶段填充此结构，供 CompilerGraph 直接消费
pub struct CompilerConstraints {
    /// 可用通用寄存器数量阈值 (普通 CPU=15, APX=31)
    pub max_gpr_count: usize,
    
    /// JIT 平铺展开的最优二维尺寸 (AMX/SME 阵列启用时扩大)
    pub optimal_tile_bits: usize,
    
    /// 是否可通过 VNNI/SVE2 直接下发 INT4 硬件解包
    pub native_int4_dot: bool,
    
    /// L1i 指令缓存哨兵: 防止大 Bucket 指令平铺颠簸
    /// 当生成代码量触达此值时 JIT 退化为 jmp
    pub l1i_budget_bytes: usize,
}
```

### 9.8 存储/网络级感知传感器结构 (DATA-MEMORY-NETWORK-SENSORS)

> **关联架构**: 02-ARCHITECTURE.md §12.6 (RDMA Pipelining 融合约束)

```rust
/// 跨机/跨片物理拓扑的传感器读数
pub struct MemoryNetworkSensors {
    /// GPU L2 Cache 驻留容量 (级别锚点)
    pub l2_cache_bytes: usize,
    
    /// AMD CCD / CCX 的 L3 分区边界探测
    pub ccx_numa_topology: Option<NumaTopology>,
    
    /// TLB 条目数上限
    pub tlb_entries: usize,
    
    /// 跨机网卡带宽 (GB/s), 用于 Pipelining 约束
    pub nic_bandwidth_gbs: Option<f32>,
    
    /// 跨机 RDMA 往返延迟 (μs), 用于 Chunk 切分下限
    pub rdma_latency_us: Option<f32>,
    
    /// ARM SVE/SME 向量宽度与 ZA 阵列尺寸
    pub arm_sme_za_size: Option<usize>,
}
```

---

## 10. Zero-Overhead Unified Semantics 核心数据结构 (DATA-ZOUS)

> **关联架构**: `02-ARCHITECTURE.md §8 ARCH-ATTN-UNIFIED`
> **目标**: 统一 Attention/KV/Weight/Position 的语义契约与访问协议，同时保留原始物理载荷，禁止引入新的公共 payload 格式。

### 10.1 WeightView (DATA-WEIGHT-VIEW)

```rust
pub struct WeightView<'a> {
    pub logical_shape: Vec<usize>,
    pub quant_type: QuantType,
    pub quant_scheme: Option<QuantScheme>,
    pub base_ptr: *const u8,
    pub byte_len: usize,
    pub stride: Vec<usize>,
    pub packing: PackingDescriptor,
    pub backing: WeightBacking<'a>,
}

pub enum WeightBacking<'a> {
    SafeTensorsMmap(&'a [u8]),
    OnnxInitializer(&'a [u8]),
    GgufBlockBytes(&'a [u8]),
    DevicePtr(u64),
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `logical_shape` | `Vec<usize>` | 逻辑形状 |
| `quant_type` | `QuantType` | 权重量化格式 |
| `quant_scheme` | `Option<QuantScheme>` | 量化方案（块大小、scale/zero 规则） |
| `base_ptr` | `*const u8` | 起始指针 |
| `byte_len` | `usize` | 原始载荷长度 |
| `stride` | `Vec<usize>` | 步长 |
| `packing` | `PackingDescriptor` | packed / block-quant 描述 |
| `backing` | `WeightBacking` | 原始载荷来源 |

**铁律**:
- ✅ GGUF block bytes 原样保留
- ✅ SafeTensors 使用 mmap slice
- ✅ ONNX 使用 initializer 原始 buffer
- ❌ 禁止先转成统一 dense F32/F16 payload

### 10.2 PositionContract (DATA-POSITION-CONTRACT)

```rust
pub enum PositionContract {
    ContiguousRange { start: u32, len: usize },
    ExplicitArray { ptr: *const u32, len: usize },
    OffsetStride { offset: u32, stride: u32, len: usize },
}
```

| 变体 | 说明 | 使用场景 |
|------|------|---------|
| `ContiguousRange` | 连续位置区间 | Prefill |
| `ExplicitArray` | 显式位置数组 | Decode / 非连续 batch |
| `OffsetStride` | 等间距位置 | 特殊生成策略 |

**铁律**:
- ✅ JIT/codegen 必须直接消费 `PositionContract`
- ❌ 禁止为 RoPE 先 materialize 新的 positions tensor 再读

### 10.3 KvLayoutContract (DATA-KV-LAYOUT-CONTRACT)

```rust
pub struct KvLayoutContract {
    pub storage_kind: KvStorageKind,
    pub kv_split: KvSplitMode,
    pub layer_stride: usize,
    pub head_stride: usize,
    pub token_stride: usize,
    pub quant_type: QuantType,
    pub append_semantics: KvAppendSemantics,
}

pub enum KvStorageKind {
    DenseSeqMajor,
    HeadMajorPersistent,
    PagedHeadMajor,
}

pub enum KvSplitMode {
    SplitHalf,
    Interleaved,
}

pub enum KvAppendSemantics {
    AppendOnly,
    Overwrite,
}
```

| 字段 | 说明 |
|------|------|
| `storage_kind` | dense 临时张量 / persistent cache / paged cache |
| `kv_split` | K/V 分 half 或交错 |
| `layer_stride` | 层间步长 |
| `head_stride` | 头间步长 |
| `token_stride` | token 间步长 |
| `quant_type` | 量化格式 |
| `append_semantics` | 追加语义 |

### 10.4 KvView (DATA-KV-VIEW)

```rust
pub enum KvView {
    DenseLocal {
        k_ptr: *const u8,
        v_ptr: *const u8,
        seq_len: usize,
        kv_dim: usize,
        quant_type: QuantType,
    },
    PersistentCache {
        base_ptr: u64,
        contract: KvLayoutContract,
        layer: usize,
        visible_range: std::ops::Range<usize>,
    },
    PagedCache {
        page_table_ptr: *const u32,
        page_data_ptr: u64,
        contract: KvLayoutContract,
        layer: usize,
        visible_range: std::ops::Range<usize>,
    },
    CompositeView {
        cached: Box<KvView>,
        append: Box<KvView>,
    },
}
```

| 变体 | 说明 |
|------|------|
| `DenseLocal` | 当前层临时 K/V，prefill 直接消费 |
| `PersistentCache` | 持久 KV cache，decode 原位访问 |
| `PagedCache` | 分页 cache 视图 |
| `CompositeView` | cache + append 窗口 |

**铁律**:
- ✅ Attention kernel 的 K/V 输入必须是 `KvView`
- ❌ 禁止要求调用方先重排成 `[total_seq, kv_dim]` dense tensor

### 10.5 AttentionSemantics (DATA-ATTENTION-SEMANTICS)

```rust
pub struct AttentionSemantics {
    pub mask_mode: MaskMode,
    pub head_mode: HeadMode,
    pub scaling: ScalingMode,
    pub rope: Option<RoPEConfig>,
    pub visibility: VisibilityMode,
}

pub enum MaskMode {
    Causal,
    Bidirectional,
}

pub enum HeadMode {
    MHA,
    GQA { ratio: usize },
    MQA,
}

pub enum ScalingMode {
    InverseSqrtHeadDim,
}

pub enum VisibilityMode {
    Prefill { seq_len: usize },
    Decode { total_seq: usize },
}
```

**设计要求**:
- prefill 与 decode 必须共享同一 `AttentionSemantics`
- lower 到不同执行计划（dense consume vs cache consume）
- ❌ 禁止通过不同 op 名称（如 `MultiHeadAttention` vs `CachedGQA`）表达不同语义分支

### 10.6 AttentionProjection / AttentionConsume (DATA-ATTN-OPS)

```rust
pub struct AttentionProjection<'a> {
    pub input: TensorView<'a>,
    pub w_q: WeightView<'a>,
    pub w_k: WeightView<'a>,
    pub w_v: WeightView<'a>,
    pub positions: PositionContract,
}

pub struct AttentionConsume {
    pub q: TensorView<'static>,
    pub kv: KvView,
    pub semantics: AttentionSemantics,
}
```

**Prefill Lowering**:
- `AttentionProjection` → `QView + KEmit + VEmit`
- `AttentionConsume(Q, DenseLocal{K,V})`
- 并行 `KvAppendIntent`

**Decode Lowering**:
- `AttentionProjection` → `Q + K_new + V_new`
- `KvAppendIntent`
- `AttentionConsume(Q, PersistentCache{...})`

### 10.7 禁止的统一方式 (DATA-ZOUS-NON-GOALS)

| 禁止项 | 原因 |
|--------|------|
| GGUF → ONNX 风格权重 payload | 额外 copy + 量化信息抹平 |
| HeadMajor cache → SeqMajor dense K/V | 全量重排损失 |
| BF16/F16/quant cache → 单点常驻缓存 | 块映射膨胀 |
| Paged KV → Dense merged tensor | 破坏零拷贝 |

### 10.8 允许的统一方式 (DATA-ZOUS-GOALS)

| 允许项 | 说明 |
|--------|------|
| metadata bridge | 统一 shape/quant_type/quant 语义 |
| view abstraction | 统一访问协议，不移动 payload |
| lowering specialization | 针对不同 `View` 生成专用 kernel |
| on-the-fly transform | RoPE / 定点映射在消费时寄存器内完成 |

## 11. 物理结构一统：Unified Virtual Page 与 Dual-Track Pool (DATA-UNIFIED-PAGE)

> **关联架构**: unified-jit-architecture-master.md
> **目标**: 结束显存中按“功能”划分的隔阂，把所有驻留内存碎片归并入基于极化精度的底层物理页框中。

### 11.1 互联通信墙极化拓扑 (SystemTopology)

编译器在生成 `DeviceProfile` 时不仅持有 `NumaTopology`，更持有整机的 `SystemTopology`：

```rust
pub struct SystemTopology {
    pub numa: NumaTopology,
    pub pcie: PcieTopology,       // PCIe 世代与 P2P Switch 树
    pub rdma: RdmaCapabilities,   // IB/RoCE网络延迟常数 (RDMA_Latency)
}
```

这些参数将转化为 `DeviceProfile` 内的恒定基准，直接切断跨 NUMA/跨 PCIe Switch 的重度聚合图，将它们打散为在物理节点内自循环的局部子图。

### 11.2 大一统物理页 (Unified Virtual Page)

废弃原先按层级和作用域孤立编写的 `KvCacheConfig` 和 `MoESharedWeight`，引入系统唯一的物理原子级驻留容器：

```rust
pub struct UnifiedVirtualPage {
    pub page_id: VirtualPageId,
    pub payload_kind: PagePayloadKind,
    pub residency: MemoryResidency,
    pub quant_type: QuantType,
}

pub enum PagePayloadKind {
    KvContext,        // K/V 张量流
    ExpertWeight,     // MoE 被唤入的冷板凳专家
    PromptSystem,     // 重度复用的只读系统提示缓存
    KnowledgeRAG,     // 通过 RAG 预插槽注入的高维特征向量
}
```

无论里面装的是上下文记忆、模型权重，还是外挂 RAG 知识，对底层 `GlobalMemoryManager` 而言，它仅仅是一个占用 1MB (或设定尺寸) 的极化数组，使用同一套基于 IRR 的 LIRS 页面置换与锁定状态机。

### 11.3 双轨极化显存池 (Dual-Track Memory Pool)

显存不再以 `dtype_size` 杂乱混居，而是被硬划分为物理隔离的双轨通道（完美兼容 FP4 Blackwell 或 4-bit INT）：

```rust
pub struct DualTrackMemoryPool {
    /// 3-bit / 4-bit / FP4 的无缩放 (Scale-Free) 主数据流连续块
    pub main_pool: BlockAllocator,
    
    /// 当需要并行的 1-bit 二值化修正或二阶残差标志位时的掩码阵列
    pub _xnor_pool: BitsetAllocator,
}
```

这种双轨切割在执行跨卡 PCIe 交换 或 RDMA 多机并行 时，仅仅依靠 `RDMA_Latency` 探针触发直接传输 `main_pool` 的裸指针 (Raw Pointers)，以最小物理宽度洞穿显存带宽墙。

---

## 12. Chunked Prefill 调度数据结构 (DATA-CHUNKED-PREFILL)

> **关联架构**: 02-ARCHITECTURE.md §10 (ARCH-CHUNKED-PREFILL), hgal-scheduler-algorithm.md §8.1
> **核心职责**: 定义支撑无限大上下文交织调度的完整数据结构。

### 12.1 ChunkedScheduler 主体

| 结构 | 字段 | 说明 |
|------|------|------|
| `ChunkedScheduler` | config, prefill_queue, decode_queue | 调度器主体 |
| `PrefillRequest` | id, total_tokens, completed_tokens, pending_chunks | Prefill 请求状态 |
| `ChunkedBatch` | decode_slots, prefill_chunks | 返回的混合批次 |

### 12.2 ChunkedConfig 配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `chunk_size` | usize | 64 | Chunk 大小 (tokens) |
| `decode_slots` | usize | 8 | 每批保留的 Decode 插槽 |
| `enable_splitfuse` | bool | ~~true~~ | ⛔ 已废弃 (REQ-SCHED-007) |
| `max_chunks_per_batch` | usize | 4 | 每批最大 Chunk 数 |

---

## 13. HGAL 调度算法完整体系 (DATA-HGAL-SCHEDULING)

> **关联架构**: hgal-scheduler-algorithm.md §1-§7
> **核心职责**: 定义 Hybrid-Granularity Adaptive LRU 的全部数据结构、优先级算法、页面状态机和配置参数。

### 13.1 Gang-Aware Eviction 原则

所有的 Eviction 对象不能是单一离散 `Page`，**必须是 `SequenceGroup`**（序列组整体换出）。禁止破坏同一序列组中不同序列的 KV 数据完整性。

### 13.2 优先级计算公式

```
priority(group) = time_penalty + recency_penalty - freq_bonus - pin_bonus
```

- `time_penalty`: 基于等待时间的惩罚分（越久优先级越低）
- `recency_penalty`: 最近访问距今时长（IRR 近似）
- `freq_bonus`: 单位时间内访问频率的加分
- `pin_bonus`: 针对 Working Set 中被标记为 Protected/Pinned 的高频页予以大加分

### 13.3 页面状态机 (Page State Machine)

| 状态 | 说明 | 转换条件 |
|------|------|----------|
| **Warm** | 新分配页面的保护期 | 分配后 100ms 且访问次数 ≥ 2 后转 Protected |
| **Protected** | 热页面（类似 LIR 状态） | 超过 `working_set_window` 未访问则降级 |
| **Standby** | 候选换出页面（类似 HIR 状态） | 被选中换出 → Evicted |

**Warm-up 防护盾**: 新分配的页面在 `warmup_duration = 100ms` 且被访问 ≥ 2 次之前，**绝对不可被换出**，防止 Cache Thrashing。

### 13.4 热页自动检测 (Working Set Detection)

检测周期: 每 N 个 generation tick。

| 条件 | 说明 | 阈值 |
|------|------|------|
| 访问频率 | 单位时间内访问次数 | `access_count >= hot_threshold` |
| 最近访问 | 最后访问时间距离现在 | `now - last_access < working_set_window` |

保护解除: 热页在 `working_set_window` 时间内未被访问则解除保护。

### 13.5 CLOCK-Pro 近似实现

低成本的 LIRS 近似实现数据结构：

| 结构 | 说明 |
|------|------|
| `clock_hand` | 扫描指针位置 |
| `cold_pages` | 冷页面候选集合 |
| `hot_pages` | 热页面集合（类似 LIR） |
| `test_pages` | 测试页面（检测访问模式变化） |

**扫描算法**:
1. 从 `clock_hand` 开始扫描
2. 页面被访问: cold → hot 或保持 hot
3. 页面未被访问: hot → cold 或保持 cold
4. `clock_hand` 前进

### 13.6 完整配置参数表 (SchedulerConfig)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `page_size` | usize | 128 | 每页 token 数 |
| `total_pages` | usize | 2048 | 总页面数 |
| `max_batch` | usize | 8 | 最大批大小 |
| `max_tokens` | usize | 4096 | 最大 token 数 |
| `warmup_duration` | Duration | 100ms | Warm-up 保护期 |
| `working_set_window` | Duration | 1s | 工作集时间窗 |
| `hot_threshold` | usize | 3 | 热页访问阈值 |
| `lir_ratio` | f32 | 0.3 | LIR 页面比例 |
| `enable_clock_pro` | bool | true | 启用 CLOCK-Pro 近似 |

### 13.7 性能调优建议

| 场景 | 建议配置 | 说明 |
|------|----------|------|
| 高并发短请求 | `warmup_duration = 50ms` | 缩短保护期 |
| 长上下文推理 | `hot_threshold = 5` | 提高热页阈值 |
| 内存受限 | `lir_ratio = 0.2` | 减少 LIR 页面 |

### 13.8 验收标准

| 标准 | 目标值 | 测量方法 |
|------|--------|----------|
| 序列组整体换出 | 通过 | 测试验证 `select_victim_groups` 返回序列组 ID |
| 新页不被立即换出 | 通过 | 验证 Warm 状态页面不被选中 |
| 热页自动锁定 | 通过 | 高频访问页面变为 Protected |
| 显存碎片率 | < 5% | 性能测试验证 |
| Thrashing 率 | < 1% | 统计 swap-in 后立即 swap-out 比例 |
| 平均换出延迟 | < 10ms | 基准测试 |
| Working Set 命中率 | > 90% | 热页访问命中率 |

### 13.9 与 VLLM 的对比

| 特性 | VLLM V1 (RECOMPUTE) | HGAL 方案 |
|------|---------------------|-----------|
| 换出粒度 | 序列级 (gang) | 序列组级 (gang) |
| 抢占模式 | Recompute (重算) | Swap (换出) + Warm-up 保护 |
| 优先级 | FCFS | IRR + 频率 + 状态 |
| 新页保护 | 无 | Warm-up 期 |
| 工作集 | 无 | 自动检测 |
| Cache Thrashing | 可能 | 避免 |

---

## 14. CPU-GPU 协同 MoE 调度结构 (DATA-HETEROGENEOUS-MOE)

> **关联架构**: 02-ARCHITECTURE.md §15.3 (CPU/GPU 真正并行)
> **设计思想**: 将原本只能在 GPU 端排队拥塞的温/冷专家，在 JIT 时就硬切分出一条物理软路由，交由 CPU 端的 `AMX/AVX512` 矩阵引擎异步接管。

```rust
/// 针对被驱逐至主机端的温专家，在 CPU 侧构建的异步独立计算流栈
pub struct CpuExpertScheduler {
    /// 强制利用 NUMA 探针将该线程流 Pin 在具有最大 L3 缓存与 AMX 指令的 P-Core 上
    pub core_affinity: NumaAffinity,
    
    /// 当落入该流的 Token 触发以下 Expert ID 时，直接唤起本地计算无需回迁 GPU
    pub active_experts: Vec<ExpertId>,
    
    /// 异步计算流，与 GPU 端的 cuStream 并发推进，在 Layer 结算时统一做残差 ADD
    pub async_stream: CpuComputeStream,
}
```

---

## 15. 跨机/跨卡 RDMA KV 同步协议 (DATA-RDMA-KV-SYNC)

> **关联架构**: 02-ARCHITECTURE.md §14.5 (RDMA/PCIe 流水线预取)
> **设计思想**: 将 "注意力预瞄" 的减法运算，化身为指导底层进行 "RDMA 跨界多播" 的乘法预测，突破单机显存墙。

```rust
/// 多卡 / 多机 (Pipeline Parallelism) 下的零拷贝 KV 穿透管理器
pub struct RdmaSyncManager {
    /// 指向双轨极化显存池的主指针，用于直接跨机 DMA
    base_pool_ptr: *const DualTrackMemoryPool,
}

impl RdmaSyncManager {
    /// 由上游 Softmax 的 Epilogue 算出质心坐标后，直接触发跨机的块级抓取
    /// 此处的 `KvSideloadManager` 将在本机建立逻辑页表的占位符 (Standby)
    pub async fn issue_rdma_prefetch(
        &self,
        predicted_centroids: &[BlockIndex],
        sideload_mgr: &mut KvSideloadManager
    ) -> Result<AsyncPrefetchToken> {
        // ... 直接发起 InfiniBand/NVLink 级别的块传输
    }
}
```
