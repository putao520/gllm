# gllm 支持千问/LLM 可行性分析

## 当前架构分析

### 现有能力

| 组件 | 实现 | 说明 |
|------|------|------|
| **Model Architecture** | Encoder-only BERT | `DynamicBertModel` |
| **Attention** | Bi-directional MHA | 无 Causal Mask |
| **Pooling** | CLS/Mean/Max | 用于生成向量 |
| **Task Types** | Embedding, Reranking | 非生成式 |
| **Backend** | Wgpu/Candle/NdArray (Burn) | 多后端支持 |

### 核心模块

```
src/
├── dynamic_bert.rs     # BERT Encoder 实现
├── engine.rs           # EmbeddingEngine / RerankEngine
├── handle.rs           # Actor 模式的异步封装
├── fallback.rs         # GPU→CPU 运行时兜底 (新增)
└── pooling.rs          # CLS/Mean 池化
```

---

## 千问 LLM 架构需求

### Qwen2 模型架构

| 组件 | Qwen2 | gllm 现有 | 差距 |
|------|-------|----------|------|
| **基础架构** | Decoder-only Transformer | Encoder-only BERT | ❌ 完全不同 |
| **Attention** | Causal (Grouped-Query) | Bi-directional MHA | ❌ 需重写 |
| **Position Encoding** | RoPE (旋转位置编码) | Learned Absolute | ❌ 需新增 |
| **KV Cache** | 必须 (生成效率) | 无 | ❌ 需新增 |
| **Tokenizer** | tiktoken (BPE) | HF Tokenizer | ⚠️ 需适配 |
| **输出层** | lm_head (vocab logits) | 无 | ❌ 需新增 |
| **采样策略** | Top-p/Top-k/Temperature | 无 | ❌ 需新增 |

### 关键差异图解

```
BERT (gllm 现有):
┌───────────────────────────────────────┐
│  Input: [CLS] token1 token2 ... [SEP] │
│              ↓                        │
│  Embedding + Position (absolute)      │
│              ↓                        │
│  Encoder × N (bi-directional attn)    │
│              ↓                        │
│  Pooling (CLS token)                  │
│              ↓                        │
│  Output: [768-dim vector]             │
└───────────────────────────────────────┘

Qwen (需要新增):
┌───────────────────────────────────────┐
│  Input: token1 token2 ... tokenN      │
│              ↓                        │
│  Embedding + RoPE (rotary)            │
│              ↓                        │
│  Decoder × N (causal + GQA)           │
│       ↓    ↓    ↓                     │
│    (KV Cache for each layer)          │
│              ↓                        │
│  lm_head (vocab size logits)          │
│              ↓                        │
│  Sampling (top-p/k/temperature)       │
│              ↓                        │
│  Output: next_token_id                │
│              ↓ (loop)                 │
│  Generated: token1 token2 ... tokenM  │
└───────────────────────────────────────┘
```

---

## 实现方案对比

### 方案 A: gllm 内部扩展 (原生 Burn)

**工作量估算**: ~4-6 周

| 模块 | 开发内容 | 难度 |
|------|----------|------|
| `DynamicDecoderModel` | 新建 Decoder 架构 | 高 |
| `RotaryEmbedding` | RoPE 位置编码 | 中 |
| `GroupedQueryAttention` | GQA 注意力 | 高 |
| `KVCache` | KV 缓存管理 | 中 |
| `Sampler` | Top-p/k/Temperature | 中 |
| `GenerationEngine` | 生成循环 | 中 |
| `QuantizedWeights` | INT4/INT8 量化 (VRAM 节省) | 高 |

**优点**:
- 统一代码库，API 一致
- Burn 多后端优势保留
- 完全可控

**缺点**:
- 开发周期长
- 需要自己实现量化 (Burn 量化支持有限)
- 模型兼容性需逐一验证

### 方案 B: 集成 llama.cpp (FFI)

**工作量估算**: ~1-2 周

```rust
// 概念示例
pub struct LlamaCppBackend {
    ctx: *mut llama_context,
    model: *mut llama_model,
}

impl LlamaCppBackend {
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;
    pub fn embed(&self, text: &str) -> Result<Vec<f32>>; // 复用现有 gllm 接口
}
```

**优点**:
- 成熟稳定，广泛验证
- 原生支持 GGUF 量化 (4-bit/8-bit)
- 支持 CPU (AVX2/NEON) 和 GPU (CUDA/Metal/Vulkan)
- 社区活跃，持续更新

**缺点**:
- 外部依赖 (C++ 编译)
- FFI 边界需要 careful 管理
- 与 Burn 不共享代码

### 方案 C: 集成 candle-transformers

**工作量估算**: ~2-3 周

```rust
// 使用 candle 生态
use candle_transformers::models::qwen2::Model as Qwen2Model;
use candle_transformers::generation::LogitsProcessor;

pub struct CandleLLMBackend {
    model: Qwen2Model,
    tokenizer: Tokenizer,
    device: Device,
}
```

**优点**:
- 纯 Rust，无 FFI
- candle 社区已实现 Qwen2
- 支持 CUDA/Metal/CPU

**缺点**:
- candle-transformers 模型覆盖不完整
- 量化支持弱于 llama.cpp
- 性能略逊于 llama.cpp

---

## 推荐方案 (纯 Rust 静态编译)

> **硬性约束**: gllm 必须编译为独立可执行文件/库，不依赖外部 C/C++ 库。

### ❌ 排除方案: llama-cpp-2

llama-cpp-2 依赖 C++ 编译的 llama.cpp，**不满足纯 Rust 要求**，排除。

### ✅ 可行方案对比

| 方案 | 工作量 | 纯 Rust | Qwen2 | 推荐度 |
|------|--------|---------|-------|--------|
| **A: Burn + Candle 混合** | 2 周 | ✅ | ✅ | ⭐⭐⭐ |
| **B: 迁移到 Candle** | 2-3 周 | ✅ | ✅ | ⭐⭐ |
| **C: Burn 原生 Decoder** | 4-6 周 | ✅ | ✅ | ⭐ |

---

### 方案 A (推荐): Burn + Candle 混合

保留现有 Burn 代码，LLM 模块使用 candle-transformers：

```toml
# Cargo.toml
[features]
default = ["embedding"]
embedding = []  # Burn-based embedding (现有)
llm = ["candle-core", "candle-nn", "candle-transformers"]

[dependencies]
# 现有 Burn 依赖
burn = { version = "0.16", features = ["wgpu", "candle", "ndarray"] }

# LLM 依赖 (可选)
candle-core = { version = "0.8", optional = true }
candle-nn = { version = "0.8", optional = true }
candle-transformers = { version = "0.8", optional = true }
```

```rust
// src/llm/mod.rs
#[cfg(feature = "llm")]
mod qwen2;
#[cfg(feature = "llm")]
mod generation;

#[cfg(feature = "llm")]
pub use generation::{GeneratorHandle, GenerationConfig};
```

**优势**:
- 纯 Rust，满足静态编译要求
- 复用 candle-transformers 已有的 Qwen2 实现
- 现有 embedding/rerank 代码无需改动
- 通过 feature flag 按需编译

**劣势**:
- 两套 ML 框架共存，编译体积增大 (~30-50MB)
- Burn 和 Candle 的 tensor 类型不兼容

### 方案 B: 全面迁移到 Candle

将现有 Burn BERT 迁移到 Candle：

```rust
// 替换
use burn::tensor::Tensor;
// 为
use candle_core::Tensor;
```

**优势**:
- 统一框架，代码一致性好
- candle 也支持 BERT embedding

**劣势**:
- 需要重写 DynamicBertModel
- Candle 的多后端支持不如 Burn 灵活

### 方案 C: Burn 原生 Decoder

在 Burn 中实现完整 Qwen2 架构：

```rust
// src/decoder/qwen2.rs
pub struct Qwen2Model<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<Qwen2DecoderLayer<B>>,
    norm: RMSNorm<B>,
    lm_head: Linear<B>,
    rotary_emb: RotaryEmbedding<B>,
}

pub struct Qwen2DecoderLayer<B: Backend> {
    self_attn: GroupedQueryAttention<B>,  // 需实现
    mlp: Qwen2MLP<B>,
    input_layernorm: RMSNorm<B>,
    post_attention_layernorm: RMSNorm<B>,
}
```

**需要实现的模块**:
- `RotaryEmbedding` (RoPE)
- `GroupedQueryAttention` (GQA)
- `RMSNorm`
- `KVCache`
- `Sampler` (top-p/k)

**优势**:
- 完全统一，无外部依赖
- 完全可控

**劣势**:
- 工作量大
- 需要逐一验证模型兼容性

---

### 短期推荐: 方案 A

```
gllm/
├── src/
│   ├── dynamic_bert.rs    # Burn BERT (保留)
│   ├── engine.rs          # Embedding/Rerank (保留)
│   ├── fallback.rs        # GPU→CPU 兜底 (已实现)
│   └── llm/               # 新增 (candle-based)
│       ├── mod.rs
│       ├── qwen2.rs       # 封装 candle-transformers
│       ├── generation.rs  # 生成循环
│       └── kv_cache.rs    # KV 缓存
```

### 中长期考虑: 方案 C

如果 Burn 社区发展出成熟的 Decoder 实现，可考虑迁移。

---

## API 设计 (推荐)

```rust
// 统一接口
pub enum GllmModel {
    /// Embedding 模型 (BERT 系列)
    Embedder(EmbedderHandle),
    /// Reranking 模型 (Cross-encoder)
    Reranker(RerankerHandle),
    /// LLM 生成模型 (新增)
    #[cfg(feature = "llm")]
    Generator(GeneratorHandle),
}

// LLM 生成接口
#[cfg(feature = "llm")]
pub struct GeneratorHandle {
    backend: LlamaBackend,
}

#[cfg(feature = "llm")]
impl GeneratorHandle {
    /// 加载模型
    pub async fn new(model_path: &str) -> Result<Self>;

    /// 生成文本
    pub async fn generate(&self, prompt: &str, config: GenerationConfig) -> Result<String>;

    /// 流式生成
    pub fn generate_stream(&self, prompt: &str, config: GenerationConfig)
        -> impl Stream<Item = Result<String>>;

    /// Chat 格式
    pub async fn chat(&self, messages: &[ChatMessage], config: GenerationConfig)
        -> Result<String>;
}

#[cfg(feature = "llm")]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub stop_sequences: Vec<String>,
}
```

---

## 资源需求对比

| 模型 | 类型 | 参数量 | VRAM (FP16) | VRAM (INT4) |
|------|------|--------|-------------|-------------|
| BGE-small-en | Embedding | 33M | ~130MB | N/A |
| GraphCodeBERT | Embedding | 125M | ~500MB | N/A |
| Qwen2-0.5B | LLM | 500M | ~1GB | ~300MB |
| Qwen2-1.5B | LLM | 1.5B | ~3GB | ~1GB |
| Qwen2-7B | LLM | 7B | ~14GB | ~4GB |
| Qwen2-72B | LLM | 72B | ~144GB | ~40GB |

**建议**: 对于本地部署，推荐 Qwen2-1.5B/7B 的 INT4 量化版本。

---

## 结论

| 维度 | 建议 |
|------|------|
| **可行性** | ✅ 完全可行（纯 Rust 方案） |
| **推荐方案** | Burn + Candle 混合（方案 A） |
| **工作量** | 2 周（混合方案） |
| **优先级** | 中等（取决于 gcode 是否需要 LLM） |
| **风险** | 低（candle-transformers 已有 Qwen2） |

**纯 Rust 优势**:
- ✅ 静态编译为独立可执行文件
- ✅ 无 C/C++ 外部依赖
- ✅ 跨平台（Linux/macOS/Windows）
- ✅ 体积可控（通过 feature flag）

**下一步行动**:
1. 确认 gcode 是否需要 LLM 能力
2. 若需要，添加 `llm` feature 并引入 candle-transformers
3. 封装 `GeneratorHandle` API
4. 支持 Qwen2-1.5B 作为首个目标（INT4 量化约 1GB VRAM）

**参考资源**:
- [candle-transformers Qwen2](https://docs.rs/candle-transformers/latest/candle_transformers/models/qwen2/index.html)
- [Crane - Pure Rust LLM Engine](https://github.com/lucasjinreal/Crane)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
