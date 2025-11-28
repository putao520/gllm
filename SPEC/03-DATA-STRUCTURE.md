# gllm 数据结构设计

## 概述

定义 gllm 库的核心数据类型和结构。

## 修订历史

| 版本 | 日期 | 描述 |
|------|------|------|
| v0.1.0 | 2025-01-28 | 初始数据结构设计 |

---

## 公共类型

### DATA-TYPE-001: Usage (使用统计)

```rust
/// Token 使用统计
#[derive(Debug, Clone, Default)]
pub struct Usage {
    /// 输入 token 数
    pub prompt_tokens: usize,
    /// 总 token 数
    pub total_tokens: usize,
}
```

### DATA-TYPE-002: Embedding (嵌入向量)

```rust
/// 嵌入向量响应
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// 嵌入向量列表
    pub embeddings: Vec<Embedding>,
    /// 使用的 token 数量
    pub usage: Usage,
}

/// 单个嵌入向量
#[derive(Debug, Clone)]
pub struct Embedding {
    /// 输入文本索引
    pub index: usize,
    /// 嵌入向量
    pub embedding: Vec<f32>,
}
```

### DATA-TYPE-003: RerankResult (重排序结果)

```rust
/// 重排序响应
#[derive(Debug, Clone)]
pub struct RerankResponse {
    /// 排序后的结果
    pub results: Vec<RerankResult>,
}

/// 单个重排序结果
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// 文档索引
    pub index: usize,
    /// 相关性分数
    pub score: f32,
    /// 原始文档 (可选)
    pub document: Option<String>,
}
```

---

## 配置类型

### DATA-CONFIG-001: ClientConfig (客户端配置)

```rust
/// 客户端配置
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// 模型存储路径 (默认 ~/.gllm/models/)
    pub models_dir: PathBuf,
    /// 设备选择
    pub device: Device,
}

/// 设备类型
#[derive(Debug, Clone, Default)]
pub enum Device {
    #[default]
    Auto,
    Gpu(usize),
    Cpu,
}
```

---

## 内部类型

### DATA-INTERNAL-001: ModelInfo (模型信息)

```rust
/// 模型元信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ModelInfo {
    /// 模型别名
    pub alias: String,
    /// HuggingFace repo ID
    pub repo_id: String,
    /// 模型类型
    pub model_type: ModelType,
    /// 模型架构
    pub architecture: Architecture,
}

/// 模型类型
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    Embedding,
    Rerank,
}

/// 模型架构
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Architecture {
    Bert,
    CrossEncoder,
}
```

### DATA-INTERNAL-002: ModelRegistry (模型注册表)

```rust
/// 模型别名注册表 (内置)
pub(crate) struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}
```

**内置模型别名**:

| 别名 | HuggingFace Repo | 类型 | 架构 |
|------|------------------|------|------|
| `bge-m3` | `BAAI/bge-m3` | Embedding | Bert |
| `bge-large-zh` | `BAAI/bge-large-zh-v1.5` | Embedding | Bert |
| `bge-small-en` | `BAAI/bge-small-en-v1.5` | Embedding | Bert |
| `bge-reranker-v2` | `BAAI/bge-reranker-v2-m3` | Rerank | CrossEncoder |
| `bge-reranker-large` | `BAAI/bge-reranker-large` | Rerank | CrossEncoder |

---

## 错误类型

### DATA-ERROR-001: Error

```rust
/// gllm 错误类型
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// 模型未找到
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// 模型下载失败
    #[error("Failed to download model: {0}")]
    DownloadError(String),

    /// 模型加载失败
    #[error("Failed to load model: {0}")]
    LoadError(String),

    /// 推理错误
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// 无效配置
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// IO 错误
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result 别名
pub type Result<T> = std::result::Result<T, Error>;
```

---

## 存储结构

### 模型目录布局

```
~/.gllm/
├── models/
│   ├── Qwen--Qwen2.5-7B/
│   │   ├── model.safetensors       # 模型权重
│   │   ├── model.safetensors.index.json  # 分片索引 (如有)
│   │   ├── config.json             # 模型配置
│   │   ├── tokenizer.json          # Tokenizer
│   │   ├── tokenizer_config.json   # Tokenizer 配置
│   │   └── special_tokens_map.json # 特殊 token 映射
│   └── BAAI--bge-m3/
│       └── ...
└── config.toml                     # 全局配置 (可选)
```

### config.json 结构 (模型配置)

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "max_position_embeddings": 32768,
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "vocab_size": 152064,
  "rope_theta": 1000000.0,
  "rms_norm_eps": 1e-6
}
```
