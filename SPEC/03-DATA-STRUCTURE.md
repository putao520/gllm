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
- gllm-kernels 负责所有反量化操作
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

### 5.1 通用配置字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `hidden_size` | u64 | Hidden dimension |
| `num_hidden_layers` | u64 | 层数 |
| `num_attention_heads` | u64 | Attention heads |
| `num_key_value_heads` | u64 | KV heads (GQA) |
| `vocab_size` | u64 | 词汇表大小 |
| `max_position_embeddings` | u64 | 最大序列长度 |
| `rope_theta` | f32 | RoPE base frequency |

### 5.2 Ω1 真实性原则配置加载

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
| U8 | `u8` | 1 |
| U16 | `u16` | 2 |
| U32 | `u32` | 4 |
| U64 | `u64` | 8 |
| BOOL | `u8` (bit-packed) | 1/8 |

---

## 5. 模型配置数据结构 (DATA-MODEL-CONFIG)

> **关联需求**: REQ-LOADER-007, REQ-LOADER-021

### 5.1 通用配置字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `hidden_size` | u64 | Hidden dimension |
| `num_hidden_layers` | u64 | 层数 |
| `num_attention_heads` | u64 | Attention heads |
| `num_key_value_heads` | u64 | KV heads (GQA) |
| `vocab_size` | u64 | 词汇表大小 |
| `max_position_embeddings` | u64 | 最大序列长度 |
| `rope_theta` | f32 | RoPE base frequency |

### 5.2 架构特定配置

| 架构 | 特殊字段 |
|------|----------|
| Llama | `sliding_window`, `attention_dropout` |
| Qwen | `use_sliding_window`, `rope_alpha` |
| Mistral | `sliding_window_pattern`, `attention_bias` |
| DeepSeek | `moe` 相关配置 |
