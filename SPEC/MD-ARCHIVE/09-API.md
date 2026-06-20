# 公共 API 设计

> **SSOT 声明**: 本文档是 gllm 客户端 API、推理接口、知识注入、Intent SDK、Guardrail、Session 复用、LoRA 适配的唯一真源。

## 1. Client 构建器模式

```rust
// 标准构建
let client = Client::builder()
    .model("Qwen/Qwen3-7B-Instruct")
    .kind(ModelKind::Chat)
    .backend(BackendType::Cpu)          // 可选：强制指定后端
    .build()?;

// 快捷方法
let chat = Client::new_chat("Qwen/Qwen3-7B-Instruct")?;
let embed = Client::new_embedding("BAAI/bge-m4")?;
let rerank = Client::new_rerank("BAAI/bge-reranker-v3")?;
```

### 1.1 Builder 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | `&str` | 是 | HuggingFace Model ID 或本地路径 |
| `kind` | `ModelKind` | 是 | `Chat` / `Embedding` / `Rerank` |
| `backend` | `BackendType` | 否 | `Cpu` / `Cuda` / `Hip` / `Metal`（默认自动检测） |
| `max_tokens` | `usize` | 否 | 最大生成 token 数 |
| `temperature` | `f32` | 否 | 采样温度 |

### 1.2 ModelKind 显式指定

`kind` 参数强制用户显式声明模型用途，消除隐式推测。

```rust
pub enum ModelKind {
    Chat,       // 文本生成（自回归）
    Embedding,  // 文本向量化（编码器）
    Rerank,     // 文本重排序（编码器 + 分类器）
}
```

## 2. 推理 API

### 2.1 文本生成

```rust
let output = client.generate("Hello, who are you?")
    .max_tokens(100)
    .temperature(0.7)
    .stream(false)
    .generate()?;

println!("{}", output.text);
```

### 2.2 批量文本嵌入

```rust
let embeddings = client.embed(vec![
    "Hello world",
    "Machine learning is fascinating"
])?;

assert_eq!(embeddings.len(), 2);
assert_eq!(embeddings[0].len(), 1024); // 维度
```

### 2.3 文本重排序

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

## 3. Tokenizer API

### 3.1 编码与解码

```rust
// 文本 → token IDs
let tokens: Vec<u32> = client.encode("Hello, world!")?;

// token IDs → 文本
let text: String = client.decode(&tokens)?;
```

模型加载时自动关联 tokenizer。`encode()` 使用模型的 tokenizer（含 special tokens 配置）。
`generate_batch()` 需要 token IDs 输入，通过 `encode()` 获取。

## 4. 批量并发推理 API (SPEC/20 BCI)

### 4.1 generate_batch()

```rust
use gllm::engine::batch_executor::{GenerateRequest, GenerateResult};

let requests = vec![
    GenerateRequest {
        request_id: 1,
        prompt_tokens: client.encode("The capital of France is")?,
        max_new_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        session_id: None,
        eos_token_id: 0,
        hook_ctx_ptr: std::ptr::null(),
        callback_table_ptr: std::ptr::null(),
    },
    // ... more requests
];

let results: Vec<GenerateResult> = client.generate_batch(&requests)?;
// results[i].output_tokens — 生成的 token IDs
// results[i].finished — 是否正常终止
```

**架构**: 单次 mega-kernel CALL 完成 M 条序列全生命周期。Rust 侧仅做 BatchContext 组装（从 Scheduler/Pager 收集页表和元数据），推理全在 JIT 内。

### 4.2 异步版本

```rust
let results: Vec<GenerateResult> = client.generate_batch_async(&requests).await?;
```

Offload 到专用线程，支持多个 batch 并发提交。

## 5. 运行时模型切换

```rust
// 原子切换（自动释放旧模型显存）
client.swap_model("Qwen/Qwen3-14B-Chat")?;

// 卸载当前模型
client.unload_model()?;

// 重新加载
client.load_model("org/model", ModelKind::Chat)?;

// 查询状态
if let Some(info) = client.model_info() {
    println!("Current model: {}", info.id);
}
```

## 4. Semantic Gatekeeper 知识注入 SDK

> **SSOT**: `SPEC/04-API-DESIGN.md §7-§8`（用户 API + 内部实现架构）+ `SPEC/SEMANTIC-GATEKEEPER.md`（技术协议）
>
> **关联需求**: `SPEC/01-REQUIREMENTS.md §12` REQ-SG-001..008

### 4.1 注册 Gatekeeper

```rust
use gllm::semantic_gatekeeper::{SemanticGatekeeperConfig, SemanticLevel};

let config = SemanticGatekeeperConfig {
    level_descriptors: [
        "struct fields and method signatures".into(),
        "validation rules and invariants".into(),
        "module dependencies and design patterns".into(),
    ],
    detection_depths: vec![0.5, 0.75, 0.9],
    gate_threshold: 0.35,
    stability_threshold: 0.95,
    alpha: 0.15,
    knowledge_provider: Arc::new(MyProvider::new()),
    ast_sentinel: Some(Arc::new(MyAstSentinel::new())),
};

client.register_semantic_gatekeeper(config)?;
```

注册后 `client.generate(...)` 自动启用 SG。取消注册用 `client.unregister_semantic_gatekeeper()`；仅清空 ActiveState 保留 Level Keys 缓存用 `client.reset_gatekeeper_state()`。

### 4.2 KnowledgeProvider Trait

```rust
pub trait KnowledgeProvider: Send + Sync {
    fn retrieve(
        &self,
        query: &[f32],
        level: SemanticLevel,
        ctx: &RetrieveContext<'_>,
    ) -> Option<KnowledgeEntry>;
}

pub struct KnowledgeEntry {
    pub text: String,       // 将被主模型 tokenizer + 冻结 embed 编码
    pub confidence: f32,    // 动态调节 α_effective = α × confidence
}
```

返回 `None` 时 SG 不注入，hidden_state 保持原样。

### 4.3 AstSentinel Trait（可选）

```rust
pub trait AstSentinel: Send + Sync {
    fn current_context<'a>(
        &self,
        generated_tokens: &'a [u32],
        tokenizer: &dyn TokenizerLookup,
    ) -> Option<AstContext<'a>>;
}
```

AST `node_kind` 变更时 SG 强制刷新 ActiveState；未注册时 SG 仅靠 hidden 锚点做稳定性判断。

> 详细契约、Level Keys 预计算、Q-tap 截获机制见 `SPEC/SEMANTIC-GATEKEEPER.md`。

## 5. Session 多轮复用 API

```rust
// 创建会话
let session = client.create_session()?;

// 第一轮
session.prompt("What is Rust?")?;
let reply1 = session.generate()?.text;

// 第二轮（复用第一轮的 KV Cache）
session.prompt("Tell me more about ownership")?;
let reply2 = session.generate()?.text;

// 结束会话
session.finish()?;
```

内部实现：Session 通过 `register_session` / `claim_session_prefix` / `finalize_session_tokens` 管理 KV Cache 确定性复用。

## 6. LoRA 动态适配 API

```rust
// 加载 LoRA 适配器
client.load_lora(
    "path/to/lora-adapter.safetensors",
    LoRAConfig {
        rank: 16,
        alpha: 32.0,             // 默认 rank as f32（LoRA 论文约定）
        target_modules: vec!["q_proj", "v_proj"],
    }
)?;

// 推理时自动使用 LoRA 权重
let output = client.generate("Translate to French: Hello")?;

// 卸载 LoRA
client.unload_lora()?;
```

### 8.1 LoRAConfig

```rust
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: Option<f32>,        // None 时默认 rank as f32
    pub target_modules: Vec<String>,
}
```

### 8.2 约束

- LoRA 适配器中的 `layer` 和 `target_module` 元数据缺失时返回错误，禁止静默降级
- 多个 LoRA 适配器可叠加（按加载顺序合并权重）
- LoRA 权重与基础权重分别存储，推理时实时融合

## 7. 错误处理

所有 API 返回 `Result<T, GllmError>`。

```rust
pub enum GllmError {
    ModelNotFound(String),
    BackendError(BackendError),
    OutOfMemory(OomHaltError),
    InvalidModelType,
    RuntimeError(String),
}
```

嵌套错误类型可通过 `source()` 方法访问完整错误链。

## 8. 内部架构映射

| API 层 | 内部组件 | 说明 |
|--------|----------|------|
| `Client` | `Arc<ArcSwapOption<ClientState>>` | Lock-free 原子模型状态容器 |
| `Executor` | `Backend + Scheduler` | 核心推理引擎 |
| `Builder` | `Loader` | 模型加载与配置 |
| `Session` | `SessionKvCache` | KV Cache 确定性复用 |
| `KnowledgeInjection` | `KvSideloadManager` | 零拷贝知识页表管理 |
| `GuardProbe` | `LayerCallback` | 安全探针回调 |
