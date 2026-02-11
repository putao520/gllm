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
| `dtype_size` | usize | 权重数据类型字节数 (F32=4, F16/BF16=2) |
| `rope_theta` | f32 | RoPE 基频 |

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
| `dtype_size` | 从权重张量实际 dtype | 任意浮点张量 |

### 6.3 格式特定推导规则

#### 6.3.1 GGUF 格式 (已实现)

| 字段 | 推导方法 | 回退策略 |
|------|----------|----------|
| `vocab_size` | `token_embd.weight` shape 较大维度 | `{arch}.vocab_size` 元数据 |
| `head_dim` | `attn_q.weight` shape / `num_heads` | `attention.head_dim` 或 `rope.dimension_count` 元数据 |
| `dtype_size` | 检测第一个浮点张量 dtype | 必须存在，否则报错 |

**实现位置**: `src/model_config.rs::from_gguf_loader()`

#### 6.3.2 SafeTensors 格式 (已实现 - REQ-LOADER-022)

| 字段 | 推导方法 | 回退策略 |
|------|----------|----------|
| `vocab_size` | `model.embed_tokens.weight` 等 shape 较大维度 | `gllm.config` 元数据 → `config.json` |
| `head_dim` | `q_proj.weight` shape / `num_heads` | `gllm.config` 元数据 → `config.json` |
| `intermediate_size`| `mlp.gate_proj.weight` / `mlp.up_proj.weight` 输出维度 | `gllm.config` 元数据 → `config.json` |
| `dtype_size` | 检测权重张量实际 dtype | 必须存在，否则报错 |

**张量名称模式**:
- Embedding: `model.embed_tokens.weight`, `embeddings.word_embeddings.weight`, `transformer.wte.weight`
- Q Projection: `model.layers.{N}.self_attn.q_proj.weight`, `encoder.layer.{N}.attention.self.query.weight`
- MLP/FFN: `model.layers.{N}.mlp.gate_proj.weight`, `model.layers.{N}.mlp.up_proj.weight`

**实现要求**:
1. 新增 `from_safetensors_loader_tensor_driven()` 方法
2. 遍历张量列表查找 embedding/Q projection
3. 优先使用张量形状，仅在找不到时回退

#### 6.3.3 ONNX 格式 (待实现 - REQ-LOADER-023)

| 字段 | 推导方法 | 回退策略 |
|------|----------|----------|
| `vocab_size` | `embed_tokens.weight` 等 initializer shape 较大维度 | ONNX graph attributes → 外部 config.json |
| `head_dim` | `q_proj.weight` / `query.weight` initializer shape / `num_heads` | ONNX graph attributes → 外部 config.json |
| `intermediate_size`| `mlp.gate_proj.weight` 等 initializer 输出维度 | ONNX graph attributes → 外部 config.json |
| `dtype_size` | 检测 initializer dtype (FLOAT/FLOAT16/BFLOAT16) | 必须存在，否则报错 |

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
| SafeTensors | 📋 待实现 | REQ-LOADER-022 |
| ONNX | 📋 待实现 | REQ-LOADER-023 |

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
    #[deprecated = "Breaks determinism"]
    ThroughputFirst,
}
```

| 策略 | 排序键 | 约束 |
|------|--------|------|
| `StrictRequestIdOrder` | `RequestId` | 默认策略，确定性优先 |
| `FifoOrder` | `enqueue_time` | 确定性 FIFO |
| `ThroughputFirst` | 实现定义 | 非默认，不可推荐 |

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
