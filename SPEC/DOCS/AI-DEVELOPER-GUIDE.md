# gllm 开发指南

## 项目概述

gllm 是一个纯 Rust 实现的本地嵌入和重排序推理库，基于 Burn 深度学习框架，提供 OpenAI 风格 SDK API。

## 核心约束

### 纯 Rust 要求
- **必须**：所有代码纯 Rust 实现
- **必须**：支持 `cargo build --release --target x86_64-unknown-linux-musl` 静态编译
- **禁止**：C/C++ 依赖、FFI 绑定、外部动态库

### 技术栈（已确定）

| 组件 | 库 | 说明 |
|------|-----|------|
| 深度学习 | burn | 纯 Rust DL 框架 |
| 模型加载 | burn-import | SafeTensors 加载 |
| 模型下载 | hf-hub (rustls-tls) | HuggingFace 客户端 |
| Tokenizer | tokenizers | HuggingFace Tokenizers |
| 异步运行时 | tokio | 可选 async 特性 |
| 错误处理 | thiserror | 错误类型定义 |

### 后端选择

| 后端 | Feature | 说明 |
|------|---------|------|
| wgpu | `wgpu` (default) | 纯 Rust GPU (Vulkan/DX12/Metal) |
| ndarray | `cpu` | 纯 Rust CPU |

**禁止**：CUDA、Metal 原生后端（破坏纯 Rust）

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
│   ├── engine.rs        # 推理引擎
│   └── types.rs         # 公共类型
└── SPEC/                # 设计文档
```

## 开发任务分解

### Block 1: 基础框架
- `Cargo.toml`: 依赖配置和 feature flags
- `src/types.rs`: 公共类型定义
- `src/registry.rs`: 模型别名注册表
- `src/lib.rs`: 模块声明

### Block 2: 模型管理
- `src/model.rs`: 模型下载和加载逻辑

### Block 3: 推理引擎
- `src/engine.rs`: BERT Embedding + CrossEncoder Rerank

### Block 4: API 层
- `src/embeddings.rs`: EmbeddingsBuilder
- `src/rerank.rs`: RerankBuilder

### Block 5: 客户端整合
- `src/client.rs`: Client / AsyncClient
- `src/lib.rs`: 完善公共导出

## API 设计要点

### Builder 模式
```rust
// Embeddings
client.embeddings(["text1", "text2"]).generate()?;

// Rerank
client.rerank("query", ["doc1", "doc2"])
    .top_n(2)
    .return_documents(true)
    .generate()?;
```

### 错误处理
- 使用 `thiserror` 定义错误类型
- 所有公共 API 返回 `Result<T, Error>`

## 内置模型别名

| 别名 | HF Repo | 类型 |
|------|---------|------|
| `bge-m3` | `BAAI/bge-m3` | Embedding |
| `bge-large-zh` | `BAAI/bge-large-zh-v1.5` | Embedding |
| `bge-small-en` | `BAAI/bge-small-en-v1.5` | Embedding |
| `bge-reranker-v2` | `BAAI/bge-reranker-v2-m3` | Rerank |
| `bge-reranker-large` | `BAAI/bge-reranker-large` | Rerank |

## 存储路径

模型存储在 `~/.gllm/models/`，目录名为 HF repo ID（`/` 替换为 `--`）：
```
~/.gllm/models/BAAI--bge-m3/
```
