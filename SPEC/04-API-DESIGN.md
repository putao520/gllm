# gllm API 设计

> **📌 SSOT**: 本文档定义 gllm 的公共 API 接口规范。

## 1. 核心设计原则

- **Builder 模式**: 复杂配置通过 Builder 链式调用
- **同步优先**: 所有操作均为同步（CPU-bound 推理无需异步运行时，无 tokio/async/await）
- **Lock-free 状态**: `arc_swap::ArcSwapOption<ClientState>` 实现零开销原子模型切换，无 RwLock
- **显式类型**: 使用强类型枚举而非字符串魔法值
- **结果导向**: 返回 `Result<Output, Error>` 清晰表达成功/失败

## 2. 客户端 (Client)

客户端是 gllm 的主要入口，负责管理模型生命周期和请求分发。

### 2.1 初始化

```rust
// 推荐用法：自动推断架构和任务类型
let client = Client::builder()
    .model("Qwen/Qwen3-7B-Instruct")
    .kind(ModelKind::Chat)  // 显式指定用途
    .backend(BackendType::Cuda) // 可选：强制指定后端
    .build()?;

// 快捷方式
let chat_client = Client::new_chat("Qwen/Qwen3-7B-Instruct")?;
let embed_client = Client::new_embedding("BAAI/bge-m3")?;

// 多模型管线 (ARCH-MULTI-MODEL-PIPELINE)
let pipeline_client = Client::builder()
    .model("BAAI/bge-m3")                           // embedder (必需)
    .reranker("BAAI/bge-reranker-v2-m3")             // reranker (可选)
    .generator("Qwen/Qwen3-7B-Instruct")            // LLM (可选)
    .inference_mode(InferenceMode::Latency)
    .build()?;
```

### 2.2 运行时模型切换 (Runtime Switching)

支持在不重启进程的情况下切换底层模型。

```rust
// 切换到新模型 (原子操作，自动释放旧模型显存)
client.swap_model("Qwen/Qwen3-14B-Chat")?;

// 卸载当前模型 (释放资源，保持 Client 实例)
client.unload_model()?;

// 重新加载
client.load_model("org/model", ModelKind::Chat)?;

// 检查当前状态
if let Some(info) = client.model_info() {
    println!("Current model: {}", info.id);
}
```

**行为约束**:
1. `swap_model` 是原子操作，失败时回滚（尽量保留原模型，或处于无模型状态）。
2. 调用期间新请求会被阻塞或排队（取决于配置）。
3. 必须彻底释放旧模型的 KV Cache 和权重显存。

## 3. 推理接口

### 3.1 文本生成 (Generator)

```rust
let output = client.generate("Hello, who are you?")
    .max_tokens(100)
    .temperature(0.7)
    .top_k(40)
    .top_p(0.9)
    .stream(false)
    .generate()?;

println!("{}", output.text);
```

#### 3.1.1 采样参数

| 参数 | 类型 | 默认 | 语义 |
|------|------|------|------|
| `temperature` | `f32` | `0.7` | `0.0` → greedy（确定性）；`>0` → 缩放 logits 后多项式采样 |
| `top_k` | `usize` | `0` | `0` → 禁用；`k>0` → 只从 logit 值最大的 k 个候选中采样 |
| `top_p` | `f32` | `1.0` | `1.0` → 禁用；`p∈(0,1)` → 按概率降序累积截断（nucleus sampling） |

#### 3.1.2 采样语义契约 (REQ-API-SAMPLING-CONTRACT)

**所有后端（CPU / CUDA / HIP / Metal）必须遵循相同的采样语义，禁止静默降级为 argmax。**

采样路径严格按以下顺序执行：

1. **Greedy**: `temperature == 0.0` → 直接对原始 logits 取 argmax，忽略 `top_k` / `top_p`；结果确定性、可复现。
2. **Stochastic** (`temperature > 0.0`):
   a. 按原始 logit 降序，取前 `top_k` 个候选（`top_k == 0` 表示保留全部）。
   b. 对候选应用 `logit / temperature` 缩放。
   c. softmax（subtract max 数值稳定）→ 得到概率分布 `p_i`。
   d. 若 `top_p ∈ (0, 1)`：按 `p_i` 降序累积，截断至累积概率首次 `≥ top_p` 的位置，再重新归一化。
   e. 多项式采样：从 `rand::thread_rng()` 取 `r ∈ [0, 1)`，落在对应 CDF 桶内的候选即为结果。
3. **组合行为**: `top_k > 0 && top_p ∈ (0, 1)` 时，先 `top_k` 截断候选集，再 `top_p` 截断累积概率 — 两者可叠加。
4. **错误路径**: 空 logits / 所有 logit 为 `-inf` / softmax 和为零 → 返回 `BackendError`，**不得**返回默认 token 0 或静默 argmax 绕过。

**实现位置** (SSOT): `src/compat/sampling.rs::sample_logits_row`。
- CPU 后端 `src/compat/cpu_backend.rs::CpuBackend::sample_from_tensor` 直接调用该函数。
- GPU 后端 `src/compat/gpu_compile.rs::sample_logits_cpu` 将 logits DtoH 后调用该函数。

**不变式**:
- `T=0, 任意 logits` → 每次调用结果恒定（argmax）。
- `T=1, uniform logits` → 不同 token 出现概率相等（统计意义上 argmax 非唯一胜出）。
- `top_k=k` → 采样结果必落在 logit 前 k 名内。
- `top_p=p (p<1, 尖峰分布)` → 采样集中在峰值候选。

### 3.2 向量嵌入 (Embedding)

```rust
// 基础用法：纯 embed
let embeddings = client.embed(vec![
    "Hello world",
    "Machine learning is fascinating"
])?;

assert_eq!(embeddings.len(), 2);
assert_eq!(embeddings[0].len(), 1024); // 维度
```

### 3.3 重排序 (Rerank)

```rust
let scores = client.rerank(
    "What is the capital of France?",
    vec![
        "Paris is the capital of France",
        "London is in UK",
        "Berlin is in Germany"
    ]
)?;
```

### 3.4 文本分类 (Classify)

支持 encoder-based (BERT/XLM-R + classifier head) 和 decoder-based (LLM + score head) 的序列分类模型。

```rust
// 快捷方式
let client = Client::new_classifier("model-with-classifier-head")?;

// 分类
let result = client.classify(["Positive review!", "Terrible product."])?;

for pred in &result.predictions {
    println!("text[{}]: label={} score={:.4}", pred.index, pred.label_id, pred.score);
    // pred.logits — 原始 logits（所有标签）
}
```

**Encoder 分类流程**:
```
tokens → BERT Encoder → CLS hidden → Pooler Dense (tanh) → Classifier Head → logits
```

**Decoder 分类流程**:
```
tokens → Decoder Layers → Last Token Hidden → Score Head → logits
```

**Response 类型**:
```rust
pub struct ClassifyResponse {
    pub predictions: Vec<ClassificationResult>,
}

pub struct ClassificationResult {
    pub index: usize,       // 输入文本索引
    pub label_id: usize,    // 预测标签 ID (argmax)
    pub score: f32,          // 预测标签的 softmax 概率
    pub logits: Vec<f32>,    // 原始 logits（所有标签）
}
```

### 3.5 Embed+Rerank 融合管线 (REQ-PIPELINE-004)


当 Client 挂载了 reranker 模型时，embed API 支持透明内部 rerank。用户体验与普通 embed 完全一致，但返回结果已按 query 相关性重排序。

```rust
// 前提：Client 已挂载 reranker
let client = Client::builder()
    .model("BAAI/bge-m3")
    .reranker("BAAI/bge-reranker-v2-m3")
    .build()?;

// Level 2: embed + 内部 rerank（对用户透明）
let results = client.embed(vec!["doc1", "doc2", "doc3"])
    .rerank_query("What is the capital of France?")  // 触发内部 rerank
    .top_n(2)                                         // rerank 后取 top-n
    .generate()?;

// 返回的 embeddings 已按 rerank score 排序
// results.embeddings[0] 是最相关的文档的 embedding
// results.rerank_scores — 可选，附带 rerank 分数
```

**行为约束**:

1. 未设 `.rerank_query()` 时行为与普通 embed 完全一致（向后兼容）
2. 设了 `.rerank_query()` 但 Client 未挂载 reranker → 返回 `GllmError::RerankerNotLoaded`
3. Reranker 与 embedder 架构相同时，encoder 权重共享（零重复加载）
4. Reranker 与 embedder 架构不同时，各自独立 encoder forward

**内部流程**:
```
embed(texts).rerank_query(query).generate()
  → texts 分词
  → [Embedder Encoder] forward → hidden_states
  → MeanPool(hidden_states) → L2Norm → embeddings
  → [Reranker] forward(query, texts) → scores  (共享 encoder 或独立 encoder)
  → 按 scores 降序重排 embeddings
  → 截取 top_n
  → 返回 EmbeddingsResponse { embeddings, rerank_scores }
```

### 3.6 Embed+Rerank+LLM 完整 RAG 管线 (REQ-PIPELINE-005)

当 Client 同时挂载了 reranker 和 generator 时，支持一体化 RAG：embed → rerank → LLM 生成。

```rust
let client = Client::builder()
    .model("BAAI/bge-m3")
    .reranker("BAAI/bge-reranker-v2-m3")
    .generator("Qwen/Qwen3-7B-Instruct")
    .build()?;

// Level 3: embed + rerank + LLM 生成
let answer = client.embed(vec!["doc1", "doc2", "doc3"])
    .rerank_query("What is the capital of France?")
    .top_n(3)
    .generate_answer("基于以下文档回答问题")?;

// answer.text — LLM 生成的答案
// answer.sources — 被选中的 top-n 文档索引
// answer.rerank_scores — 各文档的 rerank 分数
```

**内部流程**:
```
embed(texts).rerank_query(query).top_n(n).generate_answer(prompt)
  → Level 2 流程产出 top-n embeddings + 对应原文
  → 拼接 prompt: "{system_prompt}\n\n文档:\n{top_n_docs}\n\n问题: {query}"
  → [Generator LLM] forward → sample → answer
  → 返回 RagResponse { text, sources, rerank_scores }
```

**行为约束**:
1. 未挂载 generator 时调用 `.generate_answer()` → 返回 `GllmError::GeneratorNotLoaded`
2. Generator 独立于 embedder/reranker（Decoder 架构 vs Encoder 架构），权重不共享
3. top-n 文档通过文本拼接传入 LLM（非 hidden state 注入），保持 LLM 的通用性

### 3.7 多模态生成 (Multimodal Generation) — REQ-API-MULTIMODAL

**适用范围**: Gemma 4 31B/E4B 等 vision+audio 多模态 generator。文本-only 模型 (SmolLM2、Qwen3 等) 的 `.image()` / `.audio()` 被拒为 `InvalidModelType`。

#### 3.7.1 API 表面

**注入编码器**:
```rust
use gllm::compat::multimodal::MultimodalEncoder;

let client = Client::new_chat("google/gemma-4-31B-IT")?;
client.set_multimodal_encoder(Arc::new(MySigLipUsmEncoder::new()?));
```

**附加媒体输入**:
```rust
let resp = client.generate("Describe this image.")
    .image(MediaInput::File("/path/to/cat.jpg".into()))    // 或 Base64/Raw
    .max_tokens(100)
    .temperature(0.3)
    .generate()?;
```

`MediaInput` 支持四种模式,按代价递增:
| 模式 | 使用场景 |
|------|---------|
| `MediaInput::File(PathBuf)` | 本地文件,由 encoder 负责解码 |
| `MediaInput::Base64 { data, mime_type }` | 网络上传的 base64 payload |
| `MediaInput::Raw(bytes)` | 预解码的像素 / PCM 帧 |
| `MediaInput::Url(String)` | 远程 URI,encoder 负责拉取(可能 Err::NetworkUnreachable) |

#### 3.7.2 Encoder trait (REQ-API-MULTIMODAL-ENCODER)

```rust
pub trait MultimodalEncoder: Send + Sync {
    fn encode_image(&self, media: &EncoderMedia) -> Result<MultimodalEncoded, BackendError>;
    fn encode_audio(&self, media: &EncoderMedia) -> Result<MultimodalEncoded, BackendError>;
}

pub struct MultimodalEncoded {
    pub tokens: Vec<u32>,        // 虚拟 token 序列(长度由 encoder 决定)
    pub embeddings: Vec<f32>,    // [num_tokens, hidden_size] row-major
    pub kind: MediaKind,         // Image / Audio
}
```

- `tokens.len() * hidden_size == embeddings.len()` 由 `MultimodalEncoded::validate()` 强制
- encoder 实现可选 SigLIP ViT (image) / USM Conformer (audio),或任何满足 trait 的实现

#### 3.7.3 特殊 Token ID 来源 (REQ-API-MULTIMODAL-TOKENS)

- **SSOT**: `ModelConfig::multimodal_token_ids: Option<MultimodalTokenIds>`
- 解析优先级(`model_config.rs::from_value`):
  1. `config.json` 顶层 `image_token_id` / `audio_token_id` (HF 惯例)
  2. alias `boi_token_id` / `boa_token_id` (Gemma 4 命名)
  3. 两者均缺且 `vision_config` 存在 → fallback `MultimodalTokenIds::gemma4_defaults()` (258880 / 258881)
  4. 纯文本模型 → `None`,`.image()` 调用在 `execute_generation_multimodal` 第 2 步被 `InvalidModelType` 拒绝
- **禁止硬编码**: 任何 `const IMAGE_TOKEN: u32 = 258880` 直接写到 Rust 源码均违规,必须从 config 或 manifest 读

#### 3.7.4 行为约束 (REQ-API-MULTIMODAL-BEHAVIOR)

1. `.image()` / `.audio()` 调用前未 `set_multimodal_encoder` → `ClientError::InvalidModelType` (不是 silent-drop)
2. 模型不声明 `multimodal_token_ids` → `InvalidModelType` 拒绝媒体输入
3. encoder 执行失败(文件不存在、解码异常、OOM) → 原错误包装为 `RuntimeError("vision encode failed: ...")`
4. **decoder-side fusion 未就绪时禁止 silent fallback**: 当 encoder 成功产出 embedding 序列但 `executor.generate` 当前不支持 embedding 直接注入,client 必须 `Err("multimodal decoder fusion not yet implemented")` (NO_SILENT_FALLBACK)。不允许"看似成功但 media 被默默丢弃"的虚假 API。
5. 纯文本调用(无 `.image()` / `.audio()`)永远绕过多模态 routing,与既有 `generate()` 语义 100% 一致

## 4. 错误处理

所有 API 返回 `Result<T, GllmError>`。

```rust
pub enum GllmError {
    /// 模型未找到或下载失败
    ModelNotFound(String),
    /// 后端初始化失败 (如 CUDA 不可用)
    /// 嵌套类型保留完整错误信息，可通过 source() 访问
    BackendError(BackendError),
    /// 显存不足 (嵌套类型保留 OOM 详情)
    OutOfMemory(OomHaltError),
    /// 模型类型不匹配 (如用 Embedding 模型做 Chat)
    InvalidModelType,
    /// 运行时错误
    RuntimeError(String),
}
```

### 4.1 错误源链 (Error Source Chain)

嵌套错误类型 (`BackendError`, `OomHaltError`) 可通过 `source()` 方法访问，允许调用方检查详细错误信息：

```rust
match result {
    Err(GllmError::BackendError(e)) => {
        // 可通过 source() 或直接匹配检查具体变体
        if let Some(backend_err) = err.source() {
            // backend_err 是 BackendError，可匹配 Unimplemented 等变体
        }
    }
    // ...
}
```

## 5. 内部架构映射

| API 层 | 内部组件 | 说明 |
|--------|----------|------|
| `Client` | `Arc<ArcSwapOption<ClientState>>` | Lock-free 原子模型状态容器 |
| `Executor` | `Backend + Scheduler` | 核心推理引擎 |
| `Builder` | `Loader` | 模型加载与配置 |

---

## 6. GGUF Loader API (API-GGUF)

> **定位**: gllm 内部使用的 GGUF 格式解析器，**不是通用工具库**
>
> **关联需求**: REQ-LOADER-011, REQ-LOADER-014, REQ-LOADER-019
> **关联架构**: ARCH-GGUF-PARSER

### 6.1 核心接口 (极简化)

```rust
use gllm::loader::gguf::GgufReader;

/// 打开 GGUF 文件 (内存映射)
let reader = GgufReader::open("model.gguf")?;

/// Ω1: 从元数据读取架构类型
let arch = reader.architecture()?;  // "llama"

/// Ω1: 读取元数据值（通用方法，禁止硬编码默认值）
let vocab_size = reader.get_metadata_u64(&format!("{}.vocab_size", arch))?
    .ok_or(GgufError::MissingMetadata("vocab_size"))?;

let hidden_size = reader.get_metadata_u64(&format!("{}.embedding_length", arch))?
    .ok_or(GgufError::MissingMetadata("embedding_length"))?;

/// Tokenizer tokens (修复 ARRAY[STRING] bug)
let tokens = reader.tokenizer_tokens()?;  // Vec<&str>

/// 读取 Tensor (零拷贝)
let tensor = reader.tensor("token_embd.weight")?;
println!("shape: {:?}, dtype: {:?}", tensor.shape(), tensor.dtype());

/// 获取原始字节（传递给 gllm-kernels）
let data = tensor.as_bytes();
```

### 6.2 Ω1 真实性原则 API

```rust
/// 配置从 GGUF 元数据读取（禁止硬编码默认值）
impl ModelConfig {
    pub fn from_gguf(reader: &GgufReader) -> Result<Self, GgufError> {
        let arch = reader.architecture()?;

        Ok(Self {
            vocab_size: reader.get_metadata_u64(&format!("{}.vocab_size", arch))
                .ok_or_else(|| GgufError::MissingMetadata("vocab_size"))?,
            hidden_size: reader.get_metadata_u64(&format!("{}.embedding_length", arch))
                .ok_or_else(|| GgufError::MissingMetadata("embedding_length"))?,
            // ... 其他字段类似，全部从元数据读取
        })
    }
}
```

### 6.3 与 gllm-kernels 集成

> **🚨 边界约束**: GGUF 解析器只负责文件格式解析，不负责类型适配
>
> **职责分离**:
> - GGUF 解析器 (`GgufReader`): 返回原始字节 + 类型标识符
> - 适配层 (`GgufAdapter`): 负责类型映射和 gllm-kernels 集成

```rust
use gllm::loader::gguf::GgufReader;
use gllm::loader::adapter::GgufAdapter;

// GGUF 解析器: 返回原始数据
let reader = GgufReader::open("model.gguf")?;
let tensor = reader.tensor("token_embd.weight")?;
let ggml_dtype = tensor.ggml_dtype();  // GgmlDType 枚举
let data = tensor.as_bytes(); // &[u8] 原始字节

// 适配层: 负责类型映射
let adapter = GgufAdapter::new(reader)?;
let kernel_tensor = adapter.tensor_for_kernel("token_embd.weight")?;
// 返回 KernelTensorView<'_>
```

### 6.4 适配层接口 (API-GGUF-ADAPTER)

```rust
use gllm::loader::adapter::{GgufAdapter, KernelTensorView, StorageFormat, PackedBits};

/// 创建适配器
let adapter = GgufAdapter::new(reader)?;
// 或直接从文件打开
let adapter = GgufAdapter::open("model.gguf")?;

/// 将 GGUF Tensor 转换为 gllm-kernels 格式
let kernel_tensor = adapter.tensor_for_kernel("token_embd.weight")?;

/// KernelTensorView — 零拷贝张量视图
pub struct KernelTensorView<'a> {
    pub storage_format: StorageFormat, // 物理层映射结构
    pub shape: Vec<usize>,       // 形状
    pub data: &'a [u8],          // 生命周期绑定的字节切片（零拷贝）
}

/// StorageFormat — 适配层物理源格式扩展（文件中的原始物理存储格式，JIT 根据 QuantType 生成对应内核）
pub enum StorageFormat {
    F32, F16, BF16, U8,          // 表示文件中的原始物理存储格式
    PackedU8(PackedBits),        // Int1/Int2/Int4 量化打包
}
```

### 6.5 类型映射约束

> **关联架构**: gllm-kernels ARCH-QUANT-GENERIC
>
> - GGUF 解析器使用 `GgmlDType` 枚举 (文件格式层)
> - 适配层通过 `map_storage_format()` 映射到 `StorageFormat`（运行时类型）
> - 量化内核分派通过 `ggml_dtype_to_quant_type()` 映射到 `QuantType`
> - 两层映射职责分离：`map_storage` 负责提取物理布局，`ggml_dtype_to_quant_type` 直接映射为 QuantType 驱动 JIT

| GGUF 类型 | adapter StorageFormat | QuantType | 映射位置 |
|-----------|-----------------------|-----------|----------|
| F32 | F32 | None | adapter.rs |
| F16 | F16 | None | adapter.rs |
| BF16 | BF16 | None | adapter.rs |
| Q4_0/Q4_1/Q4_K/IQ4_NL/IQ4_XS/MXFP4 | PackedU8(Int4) | Q4_0/Q4_1/Q4K/IQ4NL/IQ4XS/— | adapter.rs |
| Q2_K/IQ2_XXS/IQ2_XS/IQ2_S/TQ2_0 | PackedU8(Int2) | Q2K/IQ2XXS/IQ2XS/IQ2S/— | adapter.rs |
| IQ1_S/IQ1_M/TQ1_0 | PackedU8(Int1) | IQ1S/IQ1M/— | adapter.rs |
| Q8_0/Q8_1/Q8_K/I8 | U8 | Q8_0/Q8_1/Q8K/— | adapter.rs |
| Q3_K/Q5_0/Q5_1/Q5_K/Q6_K/IQ3_XXS/IQ3_S | UnsupportedType | Q3K/Q5_0/Q5_1/Q5K/Q6K/IQ3XXS/IQ3S | adapter.rs (注1) |

> **注1**: Q3/Q5/Q6/IQ3 系列在 `ggml_dtype_to_quant_type()` 中有 QuantType 映射，但 `map_storage_format()` 返回 `UnsupportedType`。这些类型的量化内核可用，但适配层 StorageFormat 映射尚未覆盖（需要扩展 PackedBits 或新增 StorageFormat 变体）。

### 6.6 错误处理

```rust
use gllm::loader::gguf::GgufError;

match GgufReader::open("model.gguf") {
    Ok(reader) => { /* ... */ }
    Err(GgufError::InvalidMagic(m)) => {
        eprintln!("Not a GGUF file: 0x{:08x}", m);
    }
    Err(GgufError::MissingMetadata(key)) => {
        eprintln!("Missing required metadata: {}", key);
    }
    Err(GgufError::TensorNotFound(name)) => {
        eprintln!("Tensor not found: {}", name);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

---

## 7. 意图提取与知识外挂 (API-KNOWLEDGE-INJECTION)

> **定位**: gllm 高阶商业特性（Residual Bus）开发者接口。
>
> **关联架构**: ARCH-RESIDUAL-BUS

### 7.1 Semantic Anchors (语义锚点)

摒弃死板的层号（如 `layer=15`），由引擎动态测算映射层深。
```rust
pub enum LayerTarget {
    ShallowSyntax, // 浅层词法区
    MidSemantic,   // 中层语义区
    DeepLogic,     // 深层逻辑区（爆词前夕）
}
```

### 7.2 知识图谱挂载 API

使用零拷贝页表实现万字上下文极速加载。

```rust
use gllm::engine::knowledge::{KnowledgeSource, LayerTarget};

// 将外挂知识挂入中层残差流
client.inject_knowledge(
    KnowledgeSource::from_frozen_kv("company_logs_dec_2025.kv"),
    LayerTarget::MidSemantic 
)?;
```

### 7.3 Multi-Intent 降维提取 API

物理砍断后续层以加速判别式任务。

```rust
// 仅计算至中层语义区即截断，强制返回特征向量
let intent_embedding = client.encode_intent(
    "Cancel my subscription",
    LayerTarget::MidSemantic 
)?;

// intent_embedding 直接对接外部轻量分类器
```

### 7.4 In-Flight Guardrail (飞行物理护栏)

底层挂载极小分类器，实现零延迟熔断。

```rust
use gllm::engine::guard::{GuardProbe, SafetyPolicy};

// 挂载安全头至深层，超过 95% 置信度将触发底层硬件中断
client.attach_guardrail(
    GuardProbe::from_safetensors("toxicity_classifier_v1.safetensors"),
    LayerTarget::DeepLogic,
    SafetyPolicy::HaltAndVeto { threshold: 0.95 }
)?;
```

---

## §8 知识注入内部实现架构 (API-INJECTION-IMPL)

> **关联**: unified-jit-architecture-master.md §9
> **上层 API**: §7 定义了用户侧接口（`LayerTarget` 枚举、`inject_knowledge()` 等）
> **核心使命**: 将三大物理注入形态（侧载 KV、残差硬插、多路 LoRA）在工程代码结构上抽象化。

### 8.1 `KnowledgeDataSource` Trait — 数据源的多态抽象

开发者不需要理解底层什么是 `LDG.E` 或虚拟页表，只需和纯粹的数据接口打交道：

```rust
/// 知识数据源的单一多态抽象
pub trait KnowledgeDataSource {
    /// 返回注入类型标识，供编译器分流
    fn injection_kind(&self) -> InjectionKind;
    
    /// 将数据物理化至引擎可感知的格式
    fn materialize(&self, engine: &EngineContext) -> Result<MaterializedPayload>;
}

pub enum InjectionKind {
    /// 侧载 KV：业务端传入 SSD 文件柄或网络地址（预存的 4-bit 财报）
    FrozenKvChunk,
    /// 晚期插入：上游小模型（如 BERT）算好的密实特征向量列
    LateFusionVector,
    /// 领域特征挂载：带有特定领域特征缩放因子的极小权重片
    DynamicLoRA,
}
```

### 8.2 语义锚点推断 — 引擎内部实现

> **枚举定义**: `LayerTarget` 已在 §7.1 定义，此处不重复。

引擎根据加载的模型拓扑，通过"熵分布曲线"自动标定 `LayerTarget` 的物理层号映射。

### 8.3 编译器 IR 级标准拓扑扩展 (`InjectionHook` Nodes)

当 JIT 编译器（`CompilerGraph`）遇到 `Op::InjectKnowledge` 这个特殊的 IR 节点时：
1. 自动展开成符合当前硬件设备的汇编代码（如生成 `Vector Add` 微指令）
2. 将 `source` 指针硬编码进 Mega-Kernel 的 Launch Parameters 中
3. 开发者对硬件多态**绝对无感**

### 8.4 零拷贝页表管理器 (`KvSideloadManager`)

实现超大财报 0 算力注入的核心支撑库：
- 与主系统的 `GlobalMemoryManager` 紧密咬合
- 当用户传入含有 10 万字 KV 数据的 `FrozenKvChunk` 时，**不开辟任何新显存、不执行任何 `memcpy`**
- 只负责做一件事：拦截当前 Request 的**逻辑地址页表（Logical Page Table）**，并将预存的物理页（Physical Block IDs）原样插入
- 对 LLM 后续的 Attention 算子来说，这跟自己前一秒算出来的数据存在那一模一样

### 8.5 跨并发上下文分流器 (`Injection_Routing_Table`)

在大吞吐 Continuous Batching 时，同时有 128 个请求。Request A 需要注入知识，Request B 不注入：
- Mega-Kernel 启动参数携挂 `Injection_Routing_Table`（类似 MoE 路由结构）
- Kernel 内部的 `ThreadIdx` / `BlockIdx` 走到注入节点时先查表确认身份
- 属于 Request A 的线程块加载外部张量进行累加
- 属于 Request B 的线程块在同一个时钟周期内进行空转（NOP）或旁路跳过
- 完美兼容 Batching 与个性化知识挂载

### 8.6 多模态 RAG 注入调度策略 (`InjectionScheduler`)

规范多轮对话中挂载对象的生命周期：

```rust
pub struct InjectionScheduler {
    /// 测算当前 RAG 注入对象的存活率，长时间不被访问进行异步卸载
    pub ttl_policy: Duration,
    /// 当请求到来时，快速探测是否存在需要被唤醒的休眠特征
    pub hit_rate_monitor: HitRateTracker,
}
```

---

## §9 全局引擎 API 配置 (API-GLOBAL-CONFIG)

### 9.1 量化执行配置

QuantType 从加载的权重文件自动检测，JIT 据此生成硬件原生内核。不需要用户手动指定精度。

```rust
// QuantType 从权重文件自动推导，无需用户配置：
// GGUF Q4_K → QuantType::Q4K → kquant_matmul JIT 内核
// SafeTensors F16 → QuantType::F16 → float_matmul JIT 内核
// 推理过程中无类型判断分支

let client = Client::builder()
    .model("model.gguf")  // QuantType 自动检测
    .build()?;
```

### 9.2 飞行安全护栏全局守护 (`SafetyPolicyConfig`)

在应用级统一管理跨请求的安全阈值：

```rust
pub struct SafetyPolicyConfig {
    /// 是否在运行时为所有请求挂载护栏探针
    pub global_guardrail_enabled: bool,
    
    /// 当探针发现毒性特征概率 > 该阈值时，硬件级物理阻断当前请求
    pub halt_and_veto_threshold: f32, // 默认 0.95
    
    /// 指定探针挂载的锚点深度
    pub target_layer: LayerTarget,
}
```
