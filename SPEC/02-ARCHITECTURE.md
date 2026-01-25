# gllm 架构设计

## 概述

gllm 是一个纯 Rust 本地嵌入和重排序推理库，基于 gllm-kernels 的零成本算子与权重容器，提供 OpenAI 风格 SDK API。支持 Encoder (BERT) 和 Decoder (Qwen2/Mistral) 两种架构。

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
│  └── Loader          → SafeTensors → WeightMatrix/Vector    │
├─────────────────────────────────────────────────────────────┤
│  Engine Layer                                               │
│  ├── EmbeddingEngine → BERT 编码 + Pooling                  │
│  └── RerankEngine    → Cross-Encoder 推理                   │
├─────────────────────────────────────────────────────────────┤
│  gllm-kernels Runtime Backends                              │
│  ├── CUDA/ROCm/Metal/WGPU → 运行时自动检测                   │
│  └── CPU                 → 自动回退                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 技术栈

| 组件 | 库 | 版本 | 说明 |
|------|-----|------|------|
| 算子与权重容器 | gllm-kernels | latest | 零成本算子 + WeightMatrix/Vector |
| 模型导入 | safetensors | latest | SafeTensors 解析 |
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
default = []                # Fat Binary：运行时选择后端
tokio = ["tokio"]           # 异步 API 支持
quantized = []              # 量化模型支持
gpu-quantized = ["quantized"] # GPU 量化（当前为 CPU 回退）
paged-attention = []        # 分页注意力
flash-attention = []        # FlashAttention
nccl = ["gllm-kernels/nccl"] # 分布式训练/推理
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
SafeTensors 解析权重 → WeightLoader
    │
    ▼
构建 WeightMatrix/Vector → 初始化模型 → 返回 Client
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

### ARCH-ADR-001: 移除 Burn，使用 gllm-kernels 零成本抽象

**决策**: 使用 gllm-kernels 作为算子库与权重容器

**理由**:
- 零成本抽象：WeightMatrix/Vector + 原生切片 API
- 运行时后端选择：同一二进制支持多 GPU/CPU
- 纯 Rust 实现，支持静态编译且无 Burn 依赖

### ARCH-ADR-002: 使用 wgpu 作为默认 GPU 后端

**决策**: 默认启用 wgpu 后端

**理由**:
- 纯 Rust 实现，无 C++ 依赖
- 跨平台支持 (Vulkan/DX12/Metal)
- 符合静态编译要求

### ARCH-ADR-003: 模型格式支持 SafeTensors 和 GGUF

**决策**: 支持 SafeTensors (默认) 和 GGUF (量化模型) 两种格式

**理由**:
- SafeTensors 由 safetensors crate 解析，用于 HuggingFace 全精度模型
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
- 推理过程中包含可变的 KVCache/中间缓冲，默认不保证 Send/Sync
- `EmbeddingEngine`/`RerankEngine` → `EngineBackend` → `Client` 难以跨线程共享
- 在 tokio 异步环境中无法直接跨线程共享（如 `tokio::spawn`、`Arc<Client>`）

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

### ARCH-ADR-007: 集成 gllm-kernels 运行时后端

**决策**: 使用 gllm-kernels 作为底层算子库，支持运行时后端选择

**问题背景**:
- 旧架构在编译时固定后端，用户无法运行时自动选择最优设备
- 用户无法在运行时根据设备自动选择最优后端
- 缺少针对 2M+ 超长上下文的数值稳定性优化

**架构设计**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                           gllm                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Model Layer (权重加载)                                              │
│    └── WeightMatrix/Vector 用于 SafeTensors/GGUF 加载               │
├─────────────────────────────────────────────────────────────────────┤
│  Attention Layer (causal_attention.rs)                               │
│    ├── 从 WeightMatrix 获取原生切片 &[f16]                           │
│    ├── 调用 gllm_kernels::KernelDispatcher::flash_attention()       │
│    └── 从切片创建输出 Vec                                            │
├─────────────────────────────────────────────────────────────────────┤
│  Engine Layer (engine.rs)                                            │
│    └── 使用 gllm_kernels::detect_backend() 获取运行时后端           │
└─────────────────────────────────────────────────────────────────────┘
           │
           │ 运行时调用
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        gllm-kernels                                  │
├─────────────────────────────────────────────────────────────────────┤
│  KernelDispatcher                                                    │
│    ├── 运行时检测: CUDA → ROCm → Metal → WGPU → CPU                 │
│    ├── 零成本派发: match enum + #[inline(always)]                   │
│    └── 2M 上下文优化: LogSpaceSoftmax + KahanAccumulator            │
└─────────────────────────────────────────────────────────────────────┘
```

**调用示例**:

```rust
// gllm/src/attention/causal_attention.rs
use gllm_kernels::{KernelDispatcher, FlashAttentionConfig};

impl CausalAttention {
    pub fn forward(&self, q: &[f16], k: &[f16], v: &[f16]) -> Vec<f16> {
        let mut output = vec![f16::ZERO; output_len];

        // 调用优化算子
        self.dispatcher.flash_attention(
            q, k, v,
            &mut output,
            FlashAttentionConfig {
                use_log_space_softmax: true,  // 2M 上下文
                use_kahan_accumulator: true,  // 数值稳定
                ..Default::default()
            },
        );

        output
    }
}
```

**理由**:
- **运行时选择**: 同一二进制支持所有 GPU 厂商，用户无需重新编译
- **零成本抽象**: 泛型 + enum match 无 vtable 开销
- **数值稳定**: 2M+ 上下文不会溢出或精度损失
- **职责分离**: gllm 专注模型管理，gllm-kernels 专注算子优化

**依赖关系**:

```toml
# gllm/Cargo.toml
[dependencies]
gllm-kernels = { version = "0.2", default-features = false }
```

**后端选择优先级**:
1. `GLLM_BACKEND` 环境变量（强制指定）
2. 自动检测: CUDA → ROCm → Metal → WGPU → CPU

### ARCH-ADR-008: 2M 超长上下文支持

**决策**: 所有 Attention 计算必须使用 gllm-kernels 的数值稳定算法

**问题背景**:
- 标准 Softmax 的 exp() 在长序列时会溢出
- 浮点累加误差随序列长度线性增长 O(n)
- 2M token 上下文需要特殊处理

**解决方案**:

| 问题 | 解决方案 | gllm-kernels 组件 |
|------|----------|-------------------|
| exp 溢出 | Log-Space Softmax | `LogSpaceSoftmax` |
| 累加误差 | Kahan 补偿求和 | `KahanAccumulator` |
| 超长序列 | 分层累加器 | `HierarchicalAccumulator` |
| 在线计算 | 稳定累加器 | `StableAccumulator` |

**数学保证**:
- Log-Space: 避免 exp(>709) 溢出
- Kahan: 误差从 O(n) 降至 O(1)
- 分层: 支持任意长度序列

**配置方式**:

```rust
FlashAttentionConfig {
    use_log_space_softmax: true,   // 2M 上下文必须开启
    use_kahan_accumulator: true,   // 建议开启
    ..Default::default()
}

### ARCH-ADR-009: 纯 GPU MoE 管线 🔒 FROZEN

**决策**: MoE 推理必须在纯 GPU 路径执行，禁止中间 GPU→CPU→GPU 往返

**问题背景**:
- MoE routing 输出 (expert_indices, expert_weights) 在 GPU 上计算
- 旧 API `moe_forward_gpu` 接受 host slices，强制 readback 后再上传
- 这完全抵消了 GPU routing 的优化效果
- 类型安全违规：`readback<T: KernelFloat>` 不支持 U32 类型

**架构约束** (FROZEN - 禁止违反):

| 约束ID | 约束内容 | 违规示例 |
|--------|----------|----------|
| ARCH-MOE-001 | `moe_forward_gpu_pure` 必须接受 GPU tensors | 接受 `&[u32]`/`&[f32]` host slices |
| ARCH-MOE-002 | routing→forward 必须纯 GPU 数据流 | routing 输出 readback 到 CPU |
| ARCH-MOE-003 | U32 tensor 必须有类型安全的 readback | 用 f32 读取 u32 再 `to_bits()` |
| ARCH-MOE-004 | 只在最终输出时 readback | 每层都 readback hidden states |

**正确的数据流**:

```
hidden_states (GPU)
    │
    ▼
moe_route_gpu()
    │
    ├── expert_indices_gpu (GPU, U32)
    └── expert_weights_gpu (GPU, F32)
    │
    ▼
moe_forward_gpu_pure()  ← 新 API，接受 GPU tensors
    │
    ▼
moe_output (GPU)
    │
    ▼
... 继续下一层 (保持 GPU) ...
    │
    ▼
最终输出时才 readback
```

**gllm-kernels API 变更**:

```rust
// 旧 API（保留用于需要 host 控制的场景）
fn moe_forward_gpu(
    &self,
    input: &GpuTensor,
    expert_indices: &[u32],      // host slice
    expert_weights: &[f32],      // host slice
    ...
) -> Result<(), String>;

// 新 API（符合 ARCH-MOE-001/002，纯 GPU 路径）
fn moe_forward_gpu_pure(
    &self,
    input: &GpuTensor,
    expert_indices: &GpuTensor,  // GPU tensor (U32)
    expert_weights: &GpuTensor,  // GPU tensor (F32)
    all_gate_weights: &GpuTensor,
    all_up_weights: &GpuTensor,
    all_down_weights: &GpuTensor,
    output: &mut GpuTensor,
    config: MoEForwardConfig,
) -> Result<(), String>;
```

**类型安全的 U32 readback** (ARCH-MOE-003):

```rust
// gllm-kernels Backend trait 新增方法
fn readback_u32(&self, gpu: &GpuTensor, host: &mut [u32]) -> Result<(), String>;
```

**实现要求**:

| 组件 | 修改内容 |
|------|----------|
| gllm-kernels/backend.rs | 添加 `moe_forward_gpu_pure` 方法签名 |
| gllm-kernels/backend.rs | 添加 `readback_u32` 方法 |
| gllm-kernels/wgpu | 实现 `moe_forward_gpu_pure`（内部直接使用 GPU buffers） |
| gllm/moe_layer.rs | 使用新 API，移除 routing readback + re-upload |

**向后兼容**:
- 保留旧 `moe_forward_gpu` API（用于需要 host 控制的场景）
- 新代码优先使用 `moe_forward_gpu_pure`

**验收标准**:
- 单 token MoE 推理无 GPU→CPU→GPU 往返（routing→forward 纯 GPU）
- U32 tensor readback 类型安全（无 f32/to_bits hack）
- 性能提升：减少 2 次 GPU 传输（indices + weights 不再 readback）
