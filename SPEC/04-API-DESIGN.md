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

// 返回批量特征向量，底层类型依赖 TurboQuant 配置
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

// 返回特征相关性打分（浮点标量输出，不涉及底层计算退化）
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

/// StorageFormat — 适配层物理源格式扩展（非 gllm 运行时格式，会在 Load-time 被转换为 TurboQuant 位宽）
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
> - 两层映射职责分离：`map_storage` 负责提取物理布局，`ggml_dtype_to_quant_type` 结合 `TurboQuantBits` 静态选型

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

## 7. 意图提取与知识外挂 (API-INJECTION-SDK)

> **定位**: gllm 高阶商业特性（Residual Bus）开发者接口。
>
> **关联需求**: REQ-CORE-015, REQ-CORE-016
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
).await?;
```

### 7.3 Multi-Intent 降维提取 API

物理砍断后续层以加速判别式任务。

```rust
// 仅计算至中层语义区即截断，强制返回特征向量
let intent_embedding = client.encode_intent(
    "Cancel my subscription",
    LayerTarget::MidSemantic 
).await?;

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
).await?;
```
