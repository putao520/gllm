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

### 3.8 Head Routing — 同一 generator 多头 API (REQ-HR)

> **协议 SSOT**: `SPEC/HEAD-ROUTING.md`
>
> **需求**: `SPEC/01-REQUIREMENTS.md §13` REQ-HR-001..005
>
> **定位**: 同一 `Client::new_chat(...)` 加载的 generator LLM,通过运行时 API 选择输出头形态,**不重新加载权重、不重新 JIT 编译**。与加载时的 `ModelKind` 正交——`ModelKind` 决定主 head,Head Routing 在单次 API 调用中切换输出形态。

#### 3.8.1 API 表面

```rust
use gllm::{Client, ClassifyBinaryConfig, ClassifyMultiwayConfig, LayerAnchor, PoolMode};

let client = Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct")?;

// 1. Generate head (已有 §3.1, 不变)
let text = client.generate("Hello").response()?;

// 2. Binary classify head
let score: f32 = client.classify_binary(
    "Is water wet? Answer yes or no:",
    ClassifyBinaryConfig {
        positive_token: "yes".to_string(),
        negative_token: "no".to_string(),
        temperature: 1.0,
    },
)?;
// 返回 P(yes) ∈ [0.0, 1.0]

// 3. Multiway classify head
let scores: Vec<f32> = client.classify_multiway(
    "Topic:",
    &["sports", "politics", "technology"],
    ClassifyMultiwayConfig::default(),
)?;
// 返回 [P(sports), P(politics), P(technology)], sum ≈ 1.0

// 4. Encode to mid-layer
let emb: Vec<f32> = client.encode_to_layer(
    "embedding text",
    LayerAnchor::Relative(0.5),
    PoolMode::MeanPool,
)?;
// 当前阶段: Err(HeadRoutingError::MidLayerNotSupported) - FusedGraphExecutor
// 单次前向未暴露 callback 截断路径,显式拒绝以保持 NO_SILENT_FALLBACK 合规。
// 落地依赖 FusedGraphExecutor 扩展 mid-layer exit 支持。
```

#### 3.8.2 配置类型

```rust
pub struct ClassifyBinaryConfig {
    /// positive 类别对应的 token 文本 (必须单 token 化)
    pub positive_token: String,
    /// negative 类别对应的 token 文本 (必须单 token 化)
    pub negative_token: String,
    /// softmax 温度 (> 0)。1.0 保持原始分布;< 1 锐化;> 1 平滑。
    pub temperature: f32,
}

pub struct ClassifyMultiwayConfig {
    /// softmax 温度 (> 0)
    pub temperature: f32,
}

impl Default for ClassifyMultiwayConfig {
    fn default() -> Self { Self { temperature: 1.0 } }
}

pub enum LayerAnchor {
    /// 相对深度 ∈ [0.0, 1.0]。0.0 = layer 0, 1.0 = 最后一层
    Relative(f32),
    /// 绝对层索引 (0-based)
    Absolute(usize),
}

pub enum PoolMode {
    /// 对 seq 维度求平均
    MeanPool,
    /// 取最后一个 token
    LastToken,
    /// 取第一个 token (CLS 位置)
    ClsToken,
}
```

#### 3.8.3 错误类型

```rust
pub enum HeadRoutingError {
    /// label 或 positive/negative token 无法单 token 化,或 tokenize 失败
    TokenNotFound(String),
    /// LayerAnchor::Relative 越界 [0.0, 1.0] 或 Absolute 越界 [0, num_layers)
    InvalidLayerAnchor(f32),
    /// classify_multiway 收到空 labels 切片
    EmptyLabels,
    /// 配置参数非法 (如 temperature ≤ 0)
    InvalidConfig(String),
    /// encode_to_layer 当前未被 FusedGraphExecutor 支持 (显式拒绝,非 stub)
    MidLayerNotSupported,
    /// 下游 backend/tokenizer 错误传播
    Backend(String),
}
```

#### 3.8.4 行为契约 (REQ-HR-001..005)

1. **零重载**: `classify_binary` / `classify_multiway` 复用加载时的 `FusedGraphExecutor`,不触发权重重装或 JIT 重编译。多次调用之间 `Arc::as_ptr(&client.state().backend)` 地址恒定。
2. **Tied embedding**: decoder generator (SmolLM2 / Qwen3 / Gemma 等) lm_head 与 embed_tokens 权重共享,`logit_t = h_last · embed_tokens[t]`。
3. **单 token 要求**: `positive_token` / `negative_token` / `labels[i]` 必须 tokenize 为**单个** token id,否则返回 `TokenNotFound(...)`。多 token 标签需调用方拆分或替换为单 token 同义词。
4. **显式错误**: token 未找到、温度非法、层索引越界 —— 全部返回 `Err(HeadRoutingError::...)`,禁止 silent 返回 0 / 默认值 / argmax。
5. **正交于 SG**: 未注册 Semantic Gatekeeper 时,HR 读取的 hidden 就是 forward 原始结果;注册 SG 后 HR 读取的 hidden 包含 SG 残差注入——这是**设计意图**,允许知识注入驱动分类。
6. **Mid-layer encode 实现**: `encode_to_layer` 通过 `MidLayerEncodeCallback` + `FusedGraphExecutor::run_with_callbacks` 在 anchor 层 `post_node` 触发 `ExitEarly { logits: hidden_state_as_f32 }`,由 Client 层按 `PoolMode` pool 后返回 `Vec<f32>`。不同 anchor / pool 产出不同向量,证明截断真实生效。

### 3.9 Guardrail SDK — in-flight 安全 veto 探针 (REQ-GR)

> **协议 SSOT**: `SPEC/GUARDRAIL.md`
>
> **需求**: `SPEC/01-REQUIREMENTS.md §14` REQ-GR-001..005

#### 3.9.1 API 表面

```rust
use gllm::{Client, GuardProbe, GuardProbeWeights, LayerAnchor, SafetyPolicy};

let client = Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct")?;

// 从 safetensors 加载探针
let attachment = client.attach_guardrail(
    GuardProbe::from_safetensors("toxicity_probe.safetensors"),
    LayerAnchor::Relative(0.5),
    SafetyPolicy::HaltAndVeto { threshold: 0.95 },
)?;

// 或 inline 权重 (测试 / 程序生成)
let attachment = client.attach_guardrail_inline(
    GuardProbeWeights { weight: vec![0.0; 576], bias: 0.0 },
    LayerAnchor::Relative(0.5),
    SafetyPolicy::LogOnly,
)?;

let _score = client.classify_binary(prompt, cfg)?;
if attachment.is_vetoed() {
    eprintln!("blocked: {:?}", attachment.last_veto_reason());
}

client.detach_guardrail(attachment.id)?;
```

#### 3.9.2 类型定义

```rust
pub enum GuardProbe {
    FromSafetensors { path: String },
}

pub struct GuardProbeWeights {
    pub weight: Vec<f32>,
    pub bias: f32,
}

pub enum SafetyPolicy {
    HaltAndVeto { threshold: f32 },
    LogOnly,
    SampleDowngrade { min_temperature: f32 },
}

pub struct GuardrailAttachment {
    pub id: u64,
    pub actual_layer: usize,
    pub probe_name: String,
    // ... (shared state accessors)
}

impl GuardrailAttachment {
    pub fn id(&self) -> u64;
    pub fn last_score(&self) -> Option<f32>;
    pub fn is_vetoed(&self) -> bool;
    pub fn last_veto_reason(&self) -> Option<String>;
    pub fn downgraded_temperature(&self) -> Option<f32>;
    pub fn reset(&self);
}

pub enum GuardrailError {
    ProbeLoadFailed(String),
    InvalidShape(String),
    InvalidPolicy(String),
    InvalidAnchor(String),
    NoModelLoaded,
    NotFound(u64),
    Io(String),
}
```

#### 3.9.3 行为契约 (REQ-GR-001..005)

1. **零重载**: `attach_guardrail` 仅注册到 `Client.guardrails` HashMap,不触发权重重装或 JIT 重编译。
2. **探针运行时**: `GuardrailProbeCallback::post_node` 在 anchor 层触发,计算 `score = sigmoid(W · h_last + b)`。
3. **HaltAndVeto**: `score > threshold` → ExitEarly 截断前向 + `trigger_veto(reason)` → Client 看到空 logits 返回 `Err`。
4. **LogOnly**: 只 `record_score`,不改变前向,两次 classify 分数完全相等。
5. **SampleDowngrade**: 只 `record_downgrade(min_temp)`,不中断前向,供上层采样器查询。
6. **NO_SILENT_FALLBACK**: 权重加载失败 / 策略非法 / anchor 越界 → 显式 `ClientError::RuntimeError`。

### 3.10 Intent Recall SDK — 中间层 encode 语义包装 (REQ-INTENT)

> **协议 SSOT**: `SPEC/INTENT.md`
>
> **需求**: `SPEC/01-REQUIREMENTS.md §15` REQ-INTENT-001..003

```rust
use gllm::{Client, IntentEncoding, LayerAnchor, PoolMode};

let client = Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct")?;

let intent = client.encode_intent(
    "What is the weather in Paris?",
    LayerAnchor::Relative(0.5),
    PoolMode::MeanPool,
)?;

assert_eq!(intent.dim(), 576);
println!("actual_layer = {}", intent.actual_layer);
let query_vec = intent.embedding;
```

#### 3.10.1 类型定义

```rust
pub struct IntentEncoding {
    pub embedding: Vec<f32>,
    pub actual_layer: usize,
    pub pool: PoolMode,
}

impl IntentEncoding {
    pub fn dim(&self) -> usize;
    pub fn l2_norm(&self) -> f32;
}

pub enum IntentError {
    InvalidLayerAnchor(String),
    EncodeFailed(String),
    NoModelLoaded,
}
```

#### 3.10.2 行为契约

- `encode_intent` 内部直接 delegate 到 `encode_to_layer` (DRY 铁律 — 零代码复制)。
- 相同 `text/anchor/pool` 下, `encode_intent(...).embedding` 与 `encode_to_layer(...)` 逐元素相等 (|Δ| < 1e-5)。
- `actual_layer` 即 `anchor.resolve(num_layers)` 的结果,严格 < `num_layers`。
### 3.11 CoT Reasoner — 任意 LLM 原生多步推理 (REQ-COT)

> **协议 SSOT**: `SPEC/COT-REASONER.md`
>
> **需求 SSOT**: `SPEC/01-REQUIREMENTS.md §16` REQ-COT-001..006
>
> **核心定位**: 对**任意** generator LLM 原生支持 Chain-of-Thought 推理,**不依赖模型自带 thinking_head 权重**,**不新增 Backend trait 方法**。仅通过 prompt engineering + 多轮 `Client::generate` iteration 在 Client 层 orchestrate。

#### 3.11.1 基本用法

```rust
use gllm::{Client, ReasoningMode};

let client = Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct")?;

// Manual 模式: 固定步长 + 精确 budget 控制
let answer = client
    .generate("What is 127 * 83?")
    .reasoning(ReasoningMode::Manual {
        max_reasoning_tokens: 512,
        step_count: 3,
    })
    .execute()?;

println!("Final answer: {}", answer.text);
for (i, step) in answer.reasoning_trace.iter().enumerate() {
    println!("Step {}: {}", i + 1, step);
}
println!("Stopped due to: {:?}", answer.stopped_reason);
```

```rust
// Auto 模式: 引擎自动决定步数/停止
let answer = client
    .generate("Design a REST API for a blog service")
    .reasoning(ReasoningMode::Auto {
        max_total_tokens: 2048,
        entropy_threshold: Some(0.5),
        stop_patterns: vec!["Final Answer:".into(), "In conclusion,".into()],
    })
    .execute()?;
```

#### 3.11.2 ReasoningMode

| 变体 | 字段 | 语义 |
|------|------|------|
| `Manual` | `max_reasoning_tokens: usize` | 所有 reasoning step 合计最大 token 数(不含 final answer) |
| `Manual` | `step_count: usize` | 步数 ≥ 1,引擎为每步分配约 `max_reasoning_tokens / step_count` 预算 |
| `Auto` | `max_total_tokens: usize` | 所有 step 合计上限,耗尽即停 |
| `Auto` | `entropy_threshold: Option<f32>` | `Some(t)` 启用文本启发式熵收敛检测(§5.2);`None` 禁用 |
| `Auto` | `stop_patterns: Vec<String>` | 任一子串在 chunk 中命中即停,空列表 = 禁用 pattern match |

#### 3.11.3 ReasoningTemplate

可覆写的 prompt 模板:

| 字段 | 默认值 | 语义 |
|------|--------|------|
| `system_prompt` | "You are a careful reasoner. Break problems into explicit steps..." | 整条 reasoning 开头的系统引导 |
| `step_prefix` | `"Step {n}:"` | 每步前置标记,`{n}` 替换为 1-based index |
| `final_prefix` | `"Final Answer:"` | Final answer 阶段前置标记 |
| `step_separator` | `"\n\n"` | step 间分隔符 |
| `temperature` | `0.7` | 每 step 采样温度 |
| `top_k` | `0` | top-k 采样参数 |
| `top_p` | `1.0` | top-p nucleus 采样参数 |
| `final_answer_budget` | `256` | Final answer 阶段独立 token 预算 |

#### 3.11.4 ReasoningResponse

```rust
pub struct ReasoningResponse {
    pub text: String,                       // Final answer (不含 trace)
    pub reasoning_trace: Vec<String>,       // 每 step 的原始 chunk text
    pub total_reasoning_tokens: usize,      // 所有 reasoning step 的合计 token 估算
    pub actual_steps: usize,                // == reasoning_trace.len()
    pub stopped_reason: ReasoningStopReason, // BudgetExhausted / StepCountReached / PatternMatched(s) / EntropyConverged
}
```

#### 3.11.5 错误模式

| 错误 | 触发场景 |
|------|----------|
| `ClientError::NoModelLoaded` | 调用前未加载模型 |
| `ClientError::RuntimeError("cot_reasoner: ...")` | 内部某轮 `Client::generate` 失败,原 message 包装 |
| `ClientError::RuntimeError("cot_reasoner invalid config: ...")` | `step_count == 0` / `max_reasoning_tokens == 0` / `temperature <= 0` 等 |

**禁止**: 静默返回空 `ReasoningResponse`、fallback 到单次 generate、忽略 budget 溢出。所有错误显式 propagate。

#### 3.11.6 与 thinking_budget 的区分

| 关注点 | `thinking_budget(n)` (§3.1) | `reasoning(ReasoningMode)` (本节) |
|--------|----------------------------|----------------------------------|
| 作用粒度 | 单次 generate 内 `<thinking>` token 数 | 跨多次 generate 调用的 orchestration |
| 依赖权重 | **是**(qwen3-thinking 等有 thinking_head) | **否**(任意 LLM) |
| Step 概念 | 无 | 有(Manual 指定,Auto 自适应) |
| 停止信号 | budget 耗尽 | budget / step count / pattern / entropy |

两者可独立使用,不冲突。注意: 当前 CoT Reasoner 内部 `Client::generate` 调用不开启 `thinking_budget`(保持默认 `None`),避免 reasoning trace 被模型的 `<thinking>` 二次分割。

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

## 7. Semantic Gatekeeper 用户接口 (API-SEMANTIC-GATEKEEPER)

> **定位**: gllm 零训练、纯运行时的结构化知识调度与注入 SDK。
>
> **关联协议**: `SPEC/SEMANTIC-GATEKEEPER.md`（技术协议 SSOT）
>
> **关联 Callback 集成**: `SPEC/05-OPTIMIZATIONS.md §2.9`
>
> **关联需求**: `SPEC/01-REQUIREMENTS.md §12` REQ-SG-001..008

### 7.1 注册 Semantic Gatekeeper

`Client::register_semantic_gatekeeper()` 在模型加载后注册一个 Gatekeeper 实例。注册时触发 Level Keys 预计算（见 `SPEC/SEMANTIC-GATEKEEPER.md §3`）。

```rust
use gllm::semantic_gatekeeper::{
    SemanticGatekeeperConfig, SemanticLevel, KnowledgeProvider, AstSentinel,
    KnowledgeEntry, RetrieveContext, AstContext,
};

let gatekeeper_config = SemanticGatekeeperConfig {
    // 三个层级的描述文本（默认见 §7.2 表，可覆写）
    level_descriptors: [
        "struct fields and method signatures".into(),
        "validation rules and invariants".into(),
        "module dependencies and design patterns".into(),
    ],
    // 检测层相对深度列表（如 [0.5, 0.75, 0.9]）
    detection_depths: vec![0.5, 0.75, 0.9],
    // 门控阈值 τ：best_score < τ 时不注入
    gate_threshold: 0.35,
    // 稳定性阈值：hidden 与锚点相似度 > 该值时复用缓存的 v_knowledge
    stability_threshold: 0.95,
    // 残差注入强度 α ∈ [0.1, 0.3]
    alpha: 0.15,
    // 用户实现的知识源（可返回本地 LSP 检索结果或远程服务结果）
    knowledge_provider: Arc::new(MyProvider::new()),
    // 可选：AST 哨兵（Tree-sitter 等），用于语法驱动的强制刷新
    ast_sentinel: Some(Arc::new(MyAstSentinel::new())),
};

client.register_semantic_gatekeeper(gatekeeper_config)?;
```

注册后后续所有 `client.generate(...)` 调用自动启用 SG。通过 `client.unregister_semantic_gatekeeper()` 取消注册，或 `client.reset_gatekeeper_state()` 仅清空 ActiveState（保留 Level Keys 缓存）。

### 7.2 SemanticLevel 枚举

```rust
pub enum SemanticLevel {
    /// L1: 符号签名、类型成员
    L1,
    /// L2: 接口约束、业务规则
    L2,
    /// L3: 架构分层、模块职责
    L3,
}
```

| 层级 | 默认描述文本 | 典型知识类型 |
|------|--------------|--------------|
| L1 | `"struct fields and method signatures"` | LSP `documentSymbol` 结果、类型成员列表 |
| L2 | `"validation rules and invariants"` | 接口前置条件、运行时断言、业务规则 |
| L3 | `"module dependencies and design patterns"` | 项目架构摘要、模块依赖图、设计模式约定 |

### 7.3 KnowledgeProvider Trait

```rust
pub trait KnowledgeProvider: Send + Sync {
    /// 根据 Query 向量、层级、上下文检索知识条目。
    /// 返回 None 表示无匹配（SG 不注入，保持原 hidden_state）。
    fn retrieve(
        &self,
        query: &[f32],
        level: SemanticLevel,
        ctx: &RetrieveContext<'_>,
    ) -> Option<KnowledgeEntry>;
}

pub struct KnowledgeEntry {
    /// 将被主模型 tokenizer + 冻结 embed 层编码的文本。
    /// 保证与 hidden_state 处于同一语义空间。
    pub text: String,
    /// 置信度 ∈ [0.0, 1.0]，动态调节 α_effective = α × confidence
    pub confidence: f32,
}

pub struct RetrieveContext<'a> {
    pub generated_tokens: &'a [u32],
    pub ast: Option<AstContext<'a>>,
    pub step: u64,
    pub request_id: RequestId,
}
```

### 7.4 AstSentinel Trait（可选）

```rust
pub trait AstSentinel: Send + Sync {
    fn current_context<'a>(
        &self,
        generated_tokens: &'a [u32],
        tokenizer: &dyn TokenizerLookup,
    ) -> Option<AstContext<'a>>;
}

pub struct AstContext<'a> {
    /// Tree-sitter 节点 kind（如 "member_expression" / "call_expression"）
    pub node_kind: &'a str,
    pub cursor_line: u32,
    pub cursor_column: u32,
    pub prefix: &'a str,
}
```

**触发作用**：
- AST `node_kind` 变更时 SG 强制刷新 ActiveState（§8.3）
- 未注册 AST 哨兵时，SG 仅凭 Query 向量 + hidden 锚点做稳定性判断

### 7.5 部署形态

| 形态 | KnowledgeProvider 实现 | 适用场景 |
|------|------------------------|---------|
| **完全本地** | 同进程内的 LSP 客户端 + LSH 索引 + 向量查询 | 离线、高隐私、实时补全 |
| **云端推理 + 本地知识库** | HTTP/gRPC 客户端回调到用户本地服务 | 代码隐私 + 云端大模型 + 任务级延迟容忍 |

SG 内核对部署形态无感。Provider 实现者自由选择本地 vs 远程。

> **详述**: `SPEC/SEMANTIC-GATEKEEPER.md §9`

---

## §8 Semantic Gatekeeper 内部实现架构 (API-SG-IMPL)

> **协议 SSOT**: `SPEC/SEMANTIC-GATEKEEPER.md`
>
> **上层 API**: §7 定义的用户侧接口

### 8.1 整体管线

```
┌─ 模型加载期 ────────────────────────────────────────┐
│ register_semantic_gatekeeper(config):              │
│   1. 构造 EmbedLookupOnlyGraph + KProjOnlyGraph    │
│      (ARCH-FULL-JIT 合规，CompilerGraph 小图)      │
│   2. FusedGraphExecutor::new(graph).compile() ×2   │
│   3. 对每个 detection_depth:                        │
│        预计算 [K_L1, K_L2, K_L3] 存入 LevelKeysCache│
│   4. 为检测层 FusedAttentionLayer 注入 q_tap 配置   │
│      (重新编译受影响层的 FusedGraph)                │
│   5. 注册 SemanticGatekeeperCallback 到 CallbackChain│
│      优先级 90（SPEC/05-OPTIMIZATIONS.md §8）       │
└────────────────────────────────────────────────────┘

┌─ 推理期（每 decode step，每个检测层） ──────────────┐
│ SemanticGatekeeperCallback.pre_node(ctx, node_idx):│
│   详见 SPEC/SEMANTIC-GATEKEEPER.md §2.1 和 §7.1     │
└────────────────────────────────────────────────────┘
```

### 8.2 Level Keys 预计算（挑战 2 决策 B）

通过 `FusedGraphExecutor` 执行两个小 `CompilerGraph`：

- **EmbedLookupOnlyGraph**：单个 `Gather(embed_weight)` 节点（`SPEC/08-EXECUTOR.md §4.3` ARCH-GATHER-JIT）
- **KProjOnlyGraph@layer_L**：`RmsNorm(input_layernorm_weight_@L) → Gemm(k_proj_weight_@L)`

> **ARCH-FULL-JIT + ARCH-CPU-GPU-UNIFIED 合规**：小图与主推理图共享同一 `CompilerGraph` IR，Phase 3 codegen 按 `DeviceProfile` 生成 CPU/PTX/HIP/MSL 原生代码。不引入后端特化。

`LevelKeysCache` 定义与一致性不变量见 `SPEC/SEMANTIC-GATEKEEPER.md §3.5`。

### 8.3 Q 向量截获（挑战 1 决策 B）

**FusedAttentionLayer Q-Tap 变体**：检测层的 `FusedAttentionLayer` 编译期携带 `q_tap: Some(QTapConfig)`，JIT codegen 在 `q_proj` GEMM 后尾段插入 STG 指令，将 Q 向量写入设备可见的 `GatekeeperRingBuffer`。

**关键属性**：
- 融合完整性保持（未拆子算子）
- 非检测层 `q_tap: None`，零额外指令开销
- Ring buffer 双缓冲 + atomic step_index 协议防陈旧读
- 全后端统一（x86_64 / AArch64 / PTX / HIP / MSL 均由 JIT 生成）

> **详述**: `SPEC/SEMANTIC-GATEKEEPER.md §4` + `SPEC/08-EXECUTOR.md §4.2`

### 8.4 ActiveState 稳定性追踪

跨 decode step 维护 `ActiveState { level, key_hash, anchor_hidden, v_knowledge, ast_node_kind, last_step }`。

**状态转移**：
- hidden 与 anchor 相似度 > `stability_threshold` && AST 节点未变 → **ReuseCache**（跳过 LSP 检索）
- 否则 / AST 节点变更 / 新请求 → **FullCompute**（完整流程）

> **详述**: `SPEC/SEMANTIC-GATEKEEPER.md §5`

### 8.5 残差相加注入（挑战 4 决策 B）

利用现有 `LayerContext.hidden_state: &'a mut [f32]`（`SPEC/05-OPTIMIZATIONS.md §1.2`），Callback 内部计算：

```
new_hidden_last_token = hidden_state[-1] + α × confidence × v_knowledge
```

然后返回现有 `CallbackAction::InjectHidden { data }`。

**零 API 扩展**：不改 `CallbackAction` 枚举；不改 `LayerCallback` trait 签名。

### 8.6 文本编码（挑战 3 决策 B）

`KnowledgeProvider.retrieve(...)` 返回的 `KnowledgeEntry.text` 在 Callback 内部流程：

```
tokens = main_model_tokenizer.encode(text)
embed  = EmbedLookupOnlyGraph.run(tokens)       (复用 §8.2 预编译的小图)
v_knowledge = mean_pool(embed, axis=seq)        (形状 hidden_size)
```

**同语义空间保证**：使用主模型冻结 Token Embedding，`v_knowledge` 与主推理流 hidden_state 处于同一向量空间，可直接相加无需任何投影。

### 8.7 NO_ISLAND_MODULE 强制性

注册的 Gatekeeper **必须**被真实推理路径触发。集成测试验收：

- `SemanticGatekeeperCallback.pre_node` 在 `FusedGraphExecutor::run_with_kv_cache_with_callbacks` 中真实被调用（非 `#[cfg(test)]` 路径）
- Mock Provider 返回固定文本时，生成 token 分布对比无 SG 基线存在可测量差异

详见 `SPEC/SEMANTIC-GATEKEEPER.md §8`。

### 8.8 与其他优化模块的协调

| 场景 | SG 行为 |
|------|---------|
| 与 Early Exit 共存 | SG 在 `pre_node` 执行；Early Exit 在 `post_node`。若 Early Exit 在 SG 注入后的层触发，v_knowledge 的语义影响已被累积进 logits |
| 与 Gate Skip 共存 | SG 优先级 90 > Gate Skip 60。若 SG 返回 `InjectHidden`，节点仍正常执行，下游 Gate Skip 可继续判定 |
| 与 RAG Inject 共存 | SG 优先级 90 > RAG 80。两者都返回 `InjectHidden` 时，SG 先注入到 hidden_state，RAG 的后续注入在 SG 修改后的状态上叠加 |
| 与 MoE 共存 | 检测层若同时是 MoE 层，SG 在 `pre_node` 完成注入后 MoE Dispatch 路由正常进行 |
| 与 Speculative Decoding 共存 | Draft phase 跳过深层检测（复用 `SPEC/05-OPTIMIZATIONS.md §6.2 H11` 约定）。Verify phase 完整跑 SG |

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
