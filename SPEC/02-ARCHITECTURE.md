# gllm 架构设计

## 概述

gllm 是一个纯 Rust 本地嵌入和重排序推理库，基于 Burn 深度学习框架，提供 OpenAI 风格 SDK API。支持 Encoder (BERT) 和 Decoder (Qwen2/Mistral) 两种架构。

## 修订历史

| 版本 | 日期 | 描述 |
|------|------|------|
| v0.5.0 | 2025-01-11 | 新增 Decoder 架构支持 (Qwen2/Mistral)、CodeXEmbed 代码嵌入模型 |
| v0.4.1 | 2025-01-10 | GPU 检测、OOM 恢复 |
| v0.1.0 | 2025-01-28 | 初始架构设计 |

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    gllm (Rust Crate)                        │
├─────────────────────────────────────────────────────────────┤
│  Public API Layer                                           │
│  ├── Client / AsyncClient                                   │
│  ├── EmbeddingsBuilder / RerankBuilder                      │
│  └── Types (Embedding, RerankResult, etc.)                  │
├─────────────────────────────────────────────────────────────┤
│  Model Layer                                                │
│  ├── Registry        → 别名 ↔ HF repo 映射                  │
│  ├── Downloader      → hf-hub 下载到 ~/.gllm/models/        │
│  └── Loader          → SafeTensors → Burn Module            │
├─────────────────────────────────────────────────────────────┤
│  Engine Layer                                               │
│  ├── EmbeddingEngine → BERT 编码 + Pooling                  │
│  └── RerankEngine    → Cross-Encoder 推理                   │
├─────────────────────────────────────────────────────────────┤
│  Burn Backend (feature flags)                               │
│  ├── wgpu (default)  → 纯 Rust GPU                          │
│  └── ndarray         → 纯 Rust CPU                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 技术栈

| 组件 | 库 | 版本 | 说明 |
|------|-----|------|------|
| 深度学习框架 | burn | latest | 纯 Rust DL 框架 |
| 模型导入 | burn-import | latest | SafeTensors 加载 |
| 模型下载 | hf-hub | latest | HuggingFace 客户端 (rustls) |
| Tokenizer | tokenizers | latest | HuggingFace Tokenizers |
| 异步运行时 | tokio | 1.x | 可选，async 特性 |
| 序列化 | serde | 1.x | JSON/配置序列化 |
| 错误处理 | thiserror | 2.x | 错误类型定义 |

---

## 模块设计

### ARCH-MOD-001: lib.rs (入口模块)

**职责**: 导出公共 API

**导出内容**:
- `Client`, `AsyncClient`
- `EmbeddingsBuilder`, `RerankBuilder`
- `Embedding`, `EmbeddingResponse`, `RerankResult`, `RerankResponse`
- `Error`, `Result`

### ARCH-MOD-002: client.rs (客户端模块)

**职责**: 客户端实现

**组件**:
- `Client` - 同步客户端，持有模型和引擎
- `AsyncClient` - 异步客户端 (feature = "async")
- 模型加载和初始化逻辑

### ARCH-MOD-003: embeddings.rs (Embeddings 模块)

**职责**: Embeddings API

**组件**:
- `EmbeddingsBuilder` - Builder 模式
- `EmbeddingResponse` - 响应结构

### ARCH-MOD-004: rerank.rs (Rerank 模块)

**职责**: Rerank API

**组件**:
- `RerankBuilder` - Builder 模式
- `RerankResponse` - 响应结构

### ARCH-MOD-005: model.rs (模型管理模块)

**职责**: 模型下载和加载

**组件**:
- `ModelManager` - 管理模型生命周期
- `download_model()` - 从 HF 下载
- `load_model()` - 加载 SafeTensors

### ARCH-MOD-006: registry.rs (注册表模块)

**职责**: 模型别名管理

**组件**:
- `ModelRegistry` - 别名注册表
- `ModelInfo` - 模型元信息 (类型、HF repo、架构)

### ARCH-MOD-007: engine.rs (推理引擎模块)

**职责**: 推理执行

**组件**:
- `EmbeddingEngine` - BERT 嵌入推理
- `RerankEngine` - Cross-Encoder 重排序推理

### ARCH-MOD-008: types.rs (类型定义模块)

**职责**: 公共类型

**组件**:
- `Embedding`, `EmbeddingResponse` - 嵌入类型
- `RerankResult`, `RerankResponse` - 重排序类型
- `Error` - 错误类型

---

## 目录结构

```
gllm/
├── Cargo.toml
├── src/
│   ├── lib.rs           # 公共 API 导出
│   ├── client.rs        # Client / AsyncClient
│   ├── embeddings.rs    # Embeddings API
│   ├── rerank.rs        # Rerank API
│   ├── model.rs         # 模型下载/加载
│   ├── registry.rs      # 别名注册表
│   ├── engine.rs        # 推理引擎 (BERT + CrossEncoder)
│   └── types.rs         # 公共类型
├── SPEC/                # 设计文档
├── README.md
└── LICENSE
```

---

## Feature Flags

```toml
[features]
default = ["wgpu"]
wgpu = ["burn/wgpu"]       # 纯 Rust GPU 后端 (默认)
cpu = ["burn/ndarray"]      # 纯 Rust CPU 后端
async = ["tokio"]           # 异步 API 支持
```

---

## 数据流

### 模型加载流程

```
用户调用 Client::new("bge-m3")
    │
    ▼
Registry 解析别名 → "BAAI/bge-m3"
    │
    ▼
检查 ~/.gllm/models/ 是否存在
    │
    ├── 存在 → 直接加载
    │
    └── 不存在 → hf-hub 下载 → 保存到本地
    │
    ▼
SafetensorsFileRecorder 加载权重
    │
    ▼
初始化 Burn Module → 返回 Client
```

### 推理流程 (Embeddings)

```
client.embeddings(["text1", "text2"]).generate()
    │
    ▼
Tokenizer 编码输入
    │
    ▼
EmbeddingEngine BERT 前向传播
    │
    ▼
Mean Pooling → 归一化
    │
    ▼
返回 EmbeddingResponse
```

### 推理流程 (Rerank)

```
client.rerank("query", ["doc1", "doc2"]).generate()
    │
    ▼
构建 [query, doc] pairs
    │
    ▼
Tokenizer 编码每个 pair
    │
    ▼
RerankEngine Cross-Encoder 前向传播
    │
    ▼
Sigmoid → 相关性分数
    │
    ▼
排序 → 返回 RerankResponse
```

---

## 存储结构

```
~/.gllm/
└── models/
    ├── BAAI--bge-m3/              # HF repo 名称 (/ → --)
    │   ├── model.safetensors
    │   ├── config.json
    │   └── tokenizer.json
    └── BAAI--bge-reranker-v2-m3/
        ├── model.safetensors
        └── ...
```

---

## 架构决策记录 (ADR)

### ARCH-ADR-001: 选择 Burn 作为深度学习框架

**决策**: 使用 Burn 而非 Candle 或 tch-rs

**理由**:
- Burn 是纯 Rust 实现，支持静态编译
- 内置完整的 Transformer 组件 (Embedding, MultiHeadAttention, LayerNorm 等)
- 原生支持 SafeTensors 格式

### ARCH-ADR-002: 使用 wgpu 作为默认 GPU 后端

**决策**: 默认启用 wgpu 后端

**理由**:
- 纯 Rust 实现，无 C++ 依赖
- 跨平台支持 (Vulkan/DX12/Metal)
- 符合静态编译要求

### ARCH-ADR-003: 模型格式支持 SafeTensors 和 GGUF

**决策**: 支持 SafeTensors (默认) 和 GGUF (量化模型) 两种格式

**理由**:
- SafeTensors 由 Burn 原生支持，用于 HuggingFace 全精度模型
- GGUF 通过**纯 Rust 解析器**实现（无 llama.cpp 绑定），保持纯 Rust 目标
- GGUF 支持 Q4_0/Q4_K_M/Q8_0 等量化格式，显著降低内存和提升推理速度
- HuggingFace 和 llama.cpp 生态都有大量 GGUF 量化模型

**v0.11.0 新增组件**:
- `GgufLoader` - 纯 Rust GGUF 文件解析器
- `QTensor` - 量化张量，支持多种 GGML 数据类型
- `QLinear` - 量化线性层，支持 dequantize + matmul

### ARCH-ADR-003b: 支持 AWQ 量化格式

**决策**: 支持 HuggingFace AWQ (Activation-aware Weight Quantization) 格式

**理由**:
- AWQ 是 HuggingFace 生态主流的 INT4 量化格式
- 与 SafeTensors 格式兼容，仅权重存储方式不同
- 提供比 GGUF Q4 更高的精度（通过 activation-aware scaling）
- 大量预量化模型可用（TheBloke 等发布者）

**v0.11.0 新增组件**:
- `AwqWeight` - AWQ 量化权重（qweight + scales + zeros）
- `AwqLinear` - AWQ 量化线性层，支持 per-group dequantize

### ARCH-ADR-004: 使用 rustls 作为 TLS 后端

**决策**: hf-hub 使用 rustls-tls 特性

**理由**:
- 纯 Rust TLS 实现
- 支持静态编译
- 无 OpenSSL 依赖

### ARCH-ADR-005: 三大核心功能

**决策**: 支持 Embedding、Rerank 和 Text Generation

**理由**:
- v0.5.0 已添加 Decoder 架构支持 (Qwen2/Mistral)
- 复用现有 DecoderModel 实现文本生成，无额外复杂度
- 统一 API 设计：Client.embeddings() / Client.rerank() / Client.generate()
- 满足完整的 RAG 场景需求

**v0.6.0 新增组件**:
- `GeneratorModel` - 封装 DecoderModel + LmHead
- `KVCache` - 增量解码加速
- `Sampler` - Temperature/Top-p/Top-k 采样
- `GenerationBuilder` - 生成请求构建器

### ARCH-ADR-006: Actor 模式解决线程安全问题 🔒 FROZEN

**问题背景**:
- Burn 框架的 `Param<T>` 使用 `std::cell::OnceCell`，不是 `Sync`
- 这导致 `EmbeddingEngine`/`RerankEngine` → `EngineBackend` → `Client` 都不是 Send/Sync
- 在 tokio 异步环境中无法跨线程共享（如 `tokio::spawn`、`Arc<Client>`）

**决策**: 使用 Actor 模式隔离非线程安全类型

**架构设计**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     调用方（异步环境）                            │
│  Arc<EmbedderHandle> / Arc<RerankerHandle>                      │
│  ├── 天然 Send + Sync                                           │
│  └── 只包含 mpsc::Sender（线程安全）                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ mpsc channel
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    专用推理线程（Dedicated Thread）               │
│  ├── gllm::Client（非 Send/Sync，但在单线程内使用）              │
│  ├── 接收请求 → 执行推理 → 通过 oneshot 返回结果                 │
│  └── 生命周期与 Handle 绑定                                      │
└─────────────────────────────────────────────────────────────────┘
```

**通信协议**:

```rust
// 请求类型
enum EmbedRequest {
    Embed {
        text: String,
        respond: oneshot::Sender<Result<Vec<f32>>>,
    },
    EmbedBatch {
        texts: Vec<String>,
        respond: oneshot::Sender<Result<Vec<Vec<f32>>>>,
    },
    Shutdown,
}

enum RerankRequest {
    Rerank {
        query: String,
        documents: Vec<String>,
        respond: oneshot::Sender<Result<Vec<RerankResult>>>,
    },
    Shutdown,
}

// Handle（用户持有，Send + Sync）
pub struct EmbedderHandle {
    sender: mpsc::Sender<EmbedRequest>,
}

pub struct RerankerHandle {
    sender: mpsc::Sender<RerankRequest>,
}
```

**API 设计**:

```rust
// 同步 API（无 tokio 特性）
impl EmbedderHandle {
    pub fn new() -> Result<Self>;           // 启动专用线程
    pub fn embed(&self, text: &str) -> Result<Vec<f32>>;
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

// 异步 API（tokio 特性）
impl EmbedderHandle {
    pub async fn new() -> Result<Self>;     // 启动专用线程
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}
```

**理由**:
- 彻底解决 Send/Sync 问题，无需 unsafe
- Handle 只包含 channel sender，天然线程安全
- 推理在专用线程执行，避免阻塞 tokio 运行时
- 零额外依赖（复用 tokio mpsc/oneshot）
- 简单可维护，代码量约 100-150 行

**限制**:
- 所有推理请求串行执行（单线程）
- 对于高并发场景，可扩展为 worker pool（未来优化）
