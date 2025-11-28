# gllm API 测试指南

## API 端点清单

gllm 是 Rust 库，不是 HTTP API。测试的是公共 API 接口：

### 核心 API 入口

#### 1. Client 初始化 (API-CLIENT-001)
```rust
// 同步客户端
let client = Client::new("bge-m3")?;
let client = Client::with_config("bge-m3", config)?;

// 异步客户端
let client = AsyncClient::new("bge-m3").await?;
let client = AsyncClient::with_config("bge-m3", config).await?;
```

**测试要点**:
- 模型别名解析 (bge-m3 → BAAI/bge-m3)
- 模型不存在错误处理
- 配置参数验证
- 异步初始化

#### 2. Embeddings API (API-EMB-001/002)
```rust
// 同步
let response = client
    .embeddings(["text1", "text2"])
    .generate()?;

// 异步
let response = client
    .embeddings(["text1", "text2"])
    .generate()
    .await?;
```

**测试要点**:
- 单个/批量文本处理
- 嵌入向量维度正确性
- 使用量统计 (prompt_tokens, total_tokens)
- 不同 feature flags (wgpu/cpu/async)

#### 3. Rerank API (API-RERANK-001/002)
```rust
// 同步
let response = client
    .rerank("query", ["doc1", "doc2"])
    .top_n(2)
    .return_documents(true)
    .generate()?;

// 异步
let response = client
    .rerank("query", ["doc1", "doc2"])
    .top_n(2)
    .return_documents(true)
    .generate()
    .await?;
```

**测试要点**:
- 查询和文档参数处理
- top_n 参数过滤
- return_documents 控制
- 分数排序正确性 (0.0-1.0)
- 异步 API 对等性

## 测试场景矩阵

### 正常流程测试

| 场景 | 输入 | 预期输出 | 需求覆盖 |
|------|------|---------|----------|
| 基础嵌入 | 单个文本 | 正确维度向量 | REQ-INFER-001, REQ-API-001 |
| 批量嵌入 | 多个文本 | 多个向量+索引 | REQ-INFER-001, REQ-API-001 |
| 基础重排 | 查询+多个文档 | 排序结果+分数 | REQ-INFER-002, REQ-API-001 |
| Top-N 重排 | 查询+文档+top_n | 限制数量结果 | REQ-API-001 |
| 异步嵌入 | 异步调用 | 结果等同同步 | REQ-API-003 |
| 异步重排 | 异步调用 | 结果等同同步 | REQ-API-003 |

### 错误处理测试

| 错误类型 | 触发方式 | 预期错误类型 | 需求覆盖 |
|---------|---------|-------------|----------|
| 模型不存在 | 未知模型名 | Error::ModelNotFound | REQ-API-001 |
| 下载失败 | 网络错误 | Error::DownloadError | REQ-MODEL-001 |
| 加载失败 | 损坏文件 | Error::LoadError | REQ-MODEL-003 |
| 推理错误 | 无效输入 | Error::InferenceError | REQ-INFER-001, REQ-INFER-002 |

### Feature Flag 测试

| Feature 组合 | 测试重点 | 需求覆盖 |
|-------------|---------|----------|
| wgpu (默认) | GPU 后端推理 | REQ-BACKEND-001 |
| cpu | CPU 后端推理 | REQ-BACKEND-002 |
| async | 异步 API 功能 | REQ-API-003 |
| cpu+async | CPU + 异步组合 | REQ-BACKEND-002, REQ-API-003 |

## Builder 模式测试要点

### 链式调用验证
```rust
// 验证 Builder 模式流畅性
let response = client
    .rerank("query", docs)
    .top_n(3)
    .return_documents(true)
    .generate()?;
```

### 默认值测试
```rust
// 测试未设置参数的默认行为
let response = client
    .rerank("query", docs)
    .generate()?; // 默认 top_n=None, return_documents=false
```

### 参数验证测试
```rust
// 测试边界值
let response = client
    .rerank("query", docs)
    .top_n(0)  // 应该错误
    .generate();
assert!(response.is_err());

let response = client
    .rerank("query", docs)
    .top_n(1)  // 应该正常
    .generate()?;
```

## 性能测试要点

### 内存使用验证
- 模型加载后内存占用
- 批量推理内存增长
- 模型卸载内存释放

### 推理速度验证
- 单个文本推理时间
- 批量推理效率
- 不同后端性能对比

## 并发测试要点

### 异步并发
```rust
// 并发多个推理任务
let futures: Vec<_> = (0..10).map(|i| {
    client.embeddings([format!("text {}", i)])
        .generate()
}).collect();

let results = futures::future::join_all(futures).await;
```

### 多线程安全
```rust
// 多线程共享客户端
use std::sync::Arc;
let client = Arc::new(Client::new("bge-m3")?);

let handles: Vec<_> = (0..4).map(|_| {
    let client = client.clone();
    thread::spawn(move || {
        client.embeddings(["test"]).generate()
    })
}).collect();

// 验证所有线程都能正常工作
```

## 测试数据使用规范

### 文本输入测试数据
```rust
const TEST_TEXTS: &[&str] = &[
    "Hello world",              // 短文本
    "This is a longer text with multiple sentences.", // 长文本
    "中文测试文本",              // 中文
    "🚀 emoji test 🎉",         // emoji
    "",                         // 空字符串
    "a".repeat(10000).as_str(), // 极长文本
];
```

### 文档测试数据
```rust
const TEST_QUERY: &str = "What is machine learning?";

const TEST_DOCUMENTS: &[&str] = &[
    "Machine learning is a subset of artificial intelligence.",
    "The weather is sunny today.",
    "Deep learning uses neural networks.",
    "Python is a programming language.",
];
```

## 输出验证规范

### Embedding 输出验证
```rust
fn assert_embedding_response(response: &EmbeddingResponse) {
    // 基本结构验证
    assert!(!response.embeddings.is_empty());

    // 索引连续性
    for (i, emb) in response.embeddings.iter().enumerate() {
        assert_eq!(emb.index, i as u32);
    }

    // 向量维度正确性 (假设 BGE-M3 是 1024 维)
    for emb in &response.embeddings {
        assert_eq!(emb.embedding.len(), 1024);

        // 向量值有效性 (不是 NaN 或 inf)
        for &val in &emb.embedding {
            assert!(val.is_finite());
        }
    }

    // 使用量统计
    assert!(response.usage.prompt_tokens > 0);
    assert_eq!(response.usage.total_tokens, response.usage.prompt_tokens);
}
```

### Rerank 输出验证
```rust
fn assert_rerank_response(response: &RerankResponse) {
    // 基本结构验证
    assert!(!response.results.is_empty());

    // 分数有效性 (0-1 范围)
    for result in &response.results {
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(result.index < response.results.len() as u32);
    }

    // 排序正确性 (分数递减)
    for window in response.results.windows(2) {
        assert!(window[0].score >= window[1].score);
    }

    // 索引唯一性
    let indices: std::collections::HashSet<_> = response.results
        .iter()
        .map(|r| r.index)
        .collect();
    assert_eq!(indices.len(), response.results.len());

    // Top-N 限制验证 (如果设置了)
    if let Some(top_n) = response.top_n {
        assert!(response.results.len() <= top_n);
    }
}
```