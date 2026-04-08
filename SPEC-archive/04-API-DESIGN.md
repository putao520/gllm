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
    .stream(false)
    .generate()?;

println!("{}", output.text);
```

### 3.2 向量嵌入 (Embedding)

```rust
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

### 10. 自适应推测解码 API (API-ADAPTIVE-SPEC)

> **关联架构**: 02-ARCHITECTURE.md §17 (ARCH-ADAPTIVE-SPEC)
> **学术依据**: EESD, ADEPT, Goose, EqSpec, TIDE
2025-2026
2026 论文
> **核心原则**: 騡型自身的浅层变体充当 Draft Model， 配合硬件指令级 batch 合并和各向异性推测树， 在零额外模型权重的前提下提升 decode 吷 2-3×。

> **约束**: API 忂须零额外可参数 — 所有 JIT 编译在模型加载时完成。

推理热路径零编译。

> 输出分布与目标模型精确一致 (speculative decoding 数学 exact 分布") — rejection sampling 保证 exact target distribution)。

> 3 謯 Adapter 仅用于 drafting, 不 final output — 验证步骤始终使用 full model's distribution。

> **阶段 B 自适应回退**: 连续低接受率时自动回退到标准解码。

> 4. **Batch 正确性**: EqSpec 三不变量 (per-sequence attention mask + 原子 KV commit/rollback)。

> 5. **Batch 合并**: SIMD/Warp 猧别 Compact→Execute→Scatter,消除异构 batch 的 wasted lanes。

> 6. **布局代价**: JIT Fusion Engine 用 `LayoutCost` 模型选择最小代价的融合路径。
> 7. **KV Cache**: Dense+Shadow 策略 (ADEPT), 保持 dense 布局不变。

#### 10.1 启用推测解码

```rust
let client = Client::builder()
    .model("model.gguf")
    .speculative_decoding(SamplingConfig::default())  // 启用, 默认配置
    .build()?;

// 方法级 API:
let executor = client.executor();

executor.enable_speculative_decoding(SpecTreeConfig {
    max_spine_depth: 5,
    max_branch_width: 3,
    min_pld_ngram: 2,
    ngram_table_capacity: 1024,
})?;
```

#### 10.2 推测解码配置

```rust
/// 推测解码配置
pub struct SpeculativeDecodingConfig {
    /// 是否启用推测解码 (默认 false)
    pub enabled: bool,
    /// 最大 spine 深度 (默认 5)
    pub max_spine_depth: usize,
    /// 每个 spine 节点的最大 branch 宽度 (默认 3)
    pub max_branch_width: usize,
    /// PLD n-gram 匹配的最小长度 (默认 2)
    pub min_pld_ngram: usize,
    /// N-gram 频率表的最大容量 (默认 1024)
    pub ngram_table_capacity: usize,
    /// 自适应回退的连续低接受率轮次阈值 (默认 3)
    pub max_low_acceptance_streak: u32,
    /// 回退到标准解码的接受率阈值 (默认 0.3)
    pub fallback_threshold: f32,
    /// Draft 叄体层数比例 (默认 0.33, 即使用 L/3 层变体)
    pub draft_layer_ratio: f32,
}

/// 各向异性推测树配置
pub struct SpecTreeConfig {
    pub max_spine_depth: usize,
    pub max_branch_width: usize,
    pub min_pld_ngram: usize,
    pub ngram_table_capacity: usize,
}

/// Draft Adapter 权重策略
pub enum AdapterWeightStrategy {
    /// Phase A: 共享 lm_head.weight (零额外参数)
    SharedLmHead,
    /// Phase B: 独立微调权重 (额外 ≈0.1% 模型参数)
    Finetuned {
        /// 蒸馏步数 (前 N 步用 full model logits 蒸馏)
        distill_steps: usize,
    },
}

/// 稀疏 KV 稡式 (用于 Early-Exit 场景)
pub enum SparseKvMode {
    /// Dense + Shadow KV (ADEPT, 推荐)
    DenseShadow,
    /// Per-Layer Bitmap (纯 Early-Exit)
    PerLayerBitmap,
    /// CSR Compressed (极端稀疏)
    CsrCompressed,
}
```
#### 10.3 推测解码运行时控制
```rust
/// 推测解码运行时控制 — 通过 Executor 方法访问
impl Executor<B, E> {
    /// 启用推测解码 (覆盖默认配置)
    pub fn enable_speculative_decoding(&mut self, config: SpecTreeDecodingConfig);

    /// 设置 Adapter 权重策略
    pub fn set_adapter_weight_strategy(&mut self, strategy: AdapterWeightStrategy);

    /// 设置稀疏 KV 模式 (Early-Exit 场景)
    pub fn set_sparse_kv_mode(&mut self, mode: SparseKvMode);

    /// 查询当前稀疏 KV 状态 (策略选择结果 + 运行时统计)
    pub fn sparse_kv_status(&self) -> SparseKvStatus;

    /// 查询 ShadowProjector 校准指标 (前 N 步蒸馏进度)
    pub fn shadow_projector_metrics(&self) -> ShadowProjectorMetrics;

    /// 获取推测解码指标
    pub fn spec_metrics(&self) -> SpecMetrics;

    /// 手动触发 draft-verify 娡式 (通常由 step() 自动调度)
    pub fn step_speculative(&mut self) -> ExecutorResult<()>;

    /// 查询当前接受率 (用于自适应调度)
    pub fn current_acceptance_rate(&self) -> f32;
}

/// 推测解码指标
pub struct SpecMetrics {
    /// 滑动平均接受率
    pub avg_acceptance_rate: f32,
    /// 平均每轮 accepted tokens
    pub avg_accepted_tokens: f32,
    /// 平均 draft 耗时 (μs)
    pub avg_draft_time_us: f64,
    /// 平均 verify 耗时 (μs)
    pub avg_verify_time_us: f64,
    /// 平均 shadow KV 填充耗时 (μs, Early-Exit 场景)
    pub avg_shadow_time_us: f64,
    /// SIMD batch 合并节省的 FLOPS 百分比
    pub compact_saved_flops_pct: f32,
    /// 当前推测树大小
    pub current_tree_size: usize,
    /// 当前使用的 draft 层数
    pub draft_layers_used: usize,
}

/// 稀疏 KV 运行时状态
pub struct SparseKvStatus {
    /// 当前策略
    pub current_strategy: SparseKvStrategy,
    /// 总 KV entries
    pub total_entries: usize,
    /// 空洞 entries (early-exit 缺口)
    pub hole_entries: usize,
    /// 稀疏度 (0.0 = 全有效, 1.0 = 全空洞)
    pub sparsity: f64,
    /// ShadowProjector 已校准步数
    pub calibration_progress: usize,
    /// ShadowProjector 是否已收敛
    pub projector_converged: bool,
}

/// ShadowProjector 校准指标
pub struct ShadowProjectorMetrics {
    /// 已完成的校准步数 / 总步数
    pub calibration_progress: (usize, usize),
    /// 最近一步的 L2 loss (full_kv vs shadow_kv)
    pub latest_l2_loss: f64,
    /// 当前 perplexity delta (目标 < target_perplexity_delta)
    pub perplexity_delta: f64,
    /// 目标 perplexity delta
    pub target_perplexity_delta: f64,
    /// 是否已收敛
    pub converged: bool,
    /// Projector forward 平均耗时 (μs)
    pub avg_forward_time_us: f64,
    /// Projector 参数量 (bytes)
    pub parameter_bytes: usize,
}
```

#### 10.4 SAGUARO 多 GPU 推测解码 API

> **关联架构**: 02-ARCHITECTURE.md §17.10 (SAGUARO)
> **适用条件**: ≥2 GPU

```rust
/// SAGUARO 多 GPU 配置
pub struct SaguaroConfig {
    /// Draft GPU 设备索引
    pub draft_device_id: usize,
    /// Target GPU 设备索引列表 (支持 Tensor Parallelism)
    pub target_device_ids: Vec<usize>,
    /// 几何扇出: 位置 0 的初始分支数 (默认 4)
    pub fan_out_root: usize,
    /// 缓存感知采样缩放因子 C (默认 0.5)
    pub cache_penalty: f32,
    /// 自适应回退 batch 阈值 b* (默认 4)
    pub fallback_batch_threshold: usize,
    /// 是否启用 pipeline 模式 (draft+verify 流水线重叠)
    pub pipeline_enabled: bool,
}

/// SAGUARO 性能指标
pub struct SaguaroMetrics {
    /// 平均 NCCL 通信延迟 (μs)
    pub avg_nccl_latency_us: f64,
    /// Pipeline 重叠率 (0.0-1.0)
    pub pipeline_overlap_ratio: f32,
    /// 缓存命中率
    pub cache_hit_rate: f32,
    /// 几何扇出平均接受 tokens/round
    pub avg_accepted_per_round: f32,
    /// 总加速比 vs 标准 decode
    pub speedup_vs_standard: f32,
}

/// 多 GPU 推测解码控制
impl Executor<B, E> {
    /// 配置 SAGUARO 多 GPU 模式
    /// 自动选择: 单 GPU → EESD, ≥2 GPU → SAGUARO
    pub fn configure_saguaro(&mut self, config: SaguaroConfig);

    /// 获取 SAGUARO 性能指标
    pub fn saguaro_metrics(&self) -> Option<SaguaroMetrics>;

    /// 手动触发一轮 SAGUARO draft+verify (通常由 step() 自动调度)
    pub fn step_saguaro(&mut self) -> ExecutorResult<()>;
}
```
