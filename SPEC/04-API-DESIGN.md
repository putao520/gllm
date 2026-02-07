# gllm API 设计

> **📌 SSOT**: 本文档定义 gllm 的公共 API 接口规范。

## 1. 核心设计原则

- **Builder 模式**: 复杂配置通过 Builder 链式调用
- **异步优先**: 核心 IO 接口均为 `async` (兼容同步封装)
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
    .build()
    .await?;

// 快捷方式
let chat_client = Client::new_chat("Qwen/Qwen3-7B-Instruct").await?;
let embed_client = Client::new_embedding("BAAI/bge-m3").await?;
```

### 2.2 运行时模型切换 (Runtime Switching)

支持在不重启进程的情况下切换底层模型。

```rust
// 切换到新模型 (自动释放旧模型显存)
client.swap_model("Qwen/Qwen3-14B-Chat").await?;

// 卸载当前模型 (释放资源，保持 Client 实例)
client.unload_model().await?;

// 检查当前状态
if let Some(info) = client.model_info().await {
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
    .stream(true) // 返回 Stream
    .await?;

// 流式处理
while let Some(token) = output.next().await {
    print!("{}", token);
}
```

### 3.2 向量嵌入 (Embedding)

```rust
let embeddings = client.embed(vec![
    "Hello world",
    "Machine learning is fascinating"
]).await?;

// 返回 Vec<Vec<f32>>
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
).await?;

// 返回 Vec<f32> 相关性分数
```

## 4. 错误处理

所有 API 返回 `Result<T, GllmError>`。

```rust
pub enum GllmError {
    /// 模型未找到或下载失败
    ModelNotFound(String),
    /// 后端初始化失败 (如 CUDA 不可用)
    BackendError(String),
    /// 显存不足
    OutOfMemory,
    /// 模型类型不匹配 (如用 Embedding 模型做 Chat)
    InvalidModelType,
    /// 运行时错误
    RuntimeError(String),
}
```

## 5. 内部架构映射

| API 层 | 内部组件 | 说明 |
|--------|----------|------|
| `Client` | `Arc<RwLock<Option<Executor>>>` | 线程安全的执行器容器 |
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
let dtype = tensor.dtype();  // GgmlDType 枚举
let data = tensor.as_bytes(); // &[u8] 原始字节

// 适配层: 负责类型映射
let adapter = GgufAdapter::new(reader)?;
let kernel_tensor = adapter.tensor_for_kernel("token_embd.weight")?;
// 返回 gllm_kernels::TensorData
```

### 6.4 适配层接口 (API-GGUF-ADAPTER)

```rust
use gllm::loader::adapter::{GgufAdapter, AdapterError};

/// 创建适配器
let adapter = GgufAdapter::new(reader)?;

/// 将 GGUF Tensor 转换为 gllm-kernels 格式
let kernel_tensor = adapter.tensor_for_kernel("token_embd.weight")?;

/// gllm-kernels TensorData 结构
pub struct TensorData {
    pub dtype: KernelDType,      // 映射后的类型
    pub shape: Vec<usize>,       // 形状
    pub data: *const u8,         // 原始字节指针
    pub len: usize,              // 字节长度
}
```

### 6.5 类型映射约束

> **关联架构**: gllm-kernels ARCH-QUANT-GENERIC
>
> - GGUF 解析器使用 `GgmlDType` 枚举 (文件格式层)
> - gllm-kernels 使用 `QuantizedStorage<const N: usize, const BITS: u8>` 泛型
> - 适配层负责映射，不违反泛型约束

| GGUF 类型 | gllm-kernels 类型 | 映射位置 |
|-----------|------------------|----------|
| F32 | F32Block | adapter.rs |
| Q4_0 | Q4_0Block | adapter.rs |
| Q8_0 | Q8_0Block | adapter.rs |
| Q5_K | Q5_KBlock | adapter.rs |
| ... | ... | ... |

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
