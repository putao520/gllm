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

---

## 算子体系矩阵

### ARCH-OPS-001: Backend Trait 算子清单

Backend trait 定义于 `gllm-kernels/src/backend_trait.rs`，包含 **18 个方法**：

| # | 算子方法 | 用途 | 数据类型 |
|---|----------|------|----------|
| 1 | `flash_attention()` | Flash Attention 计算 | f32/f16/bf16 |
| 2 | `paged_attention()` | 分页注意力（KV Cache） | f32/f16/bf16 |
| 3 | `softmax()` | Softmax 归一化 | f32/f16/bf16 |
| 4 | `matmul()` | 矩阵乘法 | f32/f16/bf16 |
| 5 | `rope_precompute()` | RoPE 位置编码预计算 | f32 |
| 6 | `rope_apply()` | RoPE 应用 | f32/f16/bf16 |
| 7 | `rope_apply_inplace()` | RoPE 原地应用 | f32/f16/bf16 |
| 8 | `topk()` | Top-K 采样 | f32 |
| 9 | `apply_temperature()` | 温度缩放 | f32 |
| 10 | `sample_tokens()` | Token 采样 | f32 |
| 11 | `argmax()` | 贪心解码 | f32 |
| 12 | `moe_route()` | MoE 稀疏路由 | f32 |
| 13 | `compute_routing_logits()` | MoE 路由 logits | f32 |
| 14 | `rms_norm()` | RMS 归一化 | f32/f16/bf16 |
| 15 | `rms_norm_inplace()` | RMS 归一化（原地） | f32/f16/bf16 |
| 16 | `silu()` | SiLU 激活 | f32/f16/bf16 |
| 17 | `silu_inplace()` | SiLU 激活（原地） | f32/f16/bf16 |
| 18 | `add_bias()` | 偏置添加 | f32/f16/bf16 |
| 19 | `backend_type()` | 返回后端标识 | - |

### ARCH-OPS-002: 后端实现矩阵

**覆盖率：100%** - 所有后端完整实现全部 16 个方法

| 算子 | CPU | WGPU | CUDA | ROCm | Metal | GPU加速 |
|------|:---:|:----:|:----:|:----:|:-----:|:-------:|
| flash_attention | ✅ | ✅ | ✅ | ✅ | ✅ | **GPU** |
| paged_attention | ✅ | ✅ | ✅ | ✅ | ✅ | **GPU** |
| softmax | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| matmul | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| rope_precompute | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| rope_apply | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| rope_apply_inplace | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| topk | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| apply_temperature | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| sample_tokens | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| argmax | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| moe_route | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| compute_routing_logits | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| add_bias | ✅ | ✅ | ✅ | ✅ | ✅ | CPU回退 |
| backend_type | ✅ | ✅ | ✅ | ✅ | ✅ | - |

**关键发现**：仅 2 个算子（flash_attention, paged_attention）有 GPU 优化实现，其他 13 个算子均为 CPU 回退。

### ARCH-OPS-003: gllm-kernels 独立函数

除 Backend trait 外，`gllm-kernels` 还提供以下独立函数（非 trait 方法）：

| 函数 | 用途 | 文件位置 |
|------|------|----------|
| `linear_forward()` | 线性层前向 | ops.rs |
| `layer_norm_forward()` | LayerNorm | ops.rs |
| `rms_norm_forward()` | RMSNorm | ops.rs |
| `rms_norm_inplace()` | RMSNorm 原地 | ops.rs |
| `gelu_inplace()` | GELU 激活 | ops.rs |
| `silu_inplace()` | SiLU 激活 | ops.rs |
| `moe_route()` | MoE 路由（CPU） | ops.rs |

---

## 推理类型算子映射

### ARCH-INF-001: Embeddings 推理 (DynamicBertModel)

**调用链**：`forward() → Encoder → Layers → Attention + FFN → Pool`

| 算子 | 来源 | 调用位置 | 说明 |
|------|------|----------|------|
| `linear_forward` | gllm-kernels | Q/K/V/O 投影, FFN | CPU |
| `layer_norm_forward` | gllm-kernels | 输入/输出归一化 | CPU |
| `gelu_inplace` | gllm-kernels | FFN 激活 | CPU |
| `backend.flash_attention` | Backend | 自注意力 | **可GPU** |
| `rope_apply_inplace` | gllm-kernels | 位置编码 | CPU, 可选 |

### ARCH-INF-002: Reranker 推理 (DynamicCrossEncoder)

**调用链**：`score() → Encoder.forward() → Pool → Classifier`

| 算子 | 来源 | 调用位置 | 说明 |
|------|------|----------|------|
| `linear_forward` | gllm-kernels | Encoder + Classifier | CPU |
| `layer_norm_forward` | gllm-kernels | Encoder 归一化 | CPU |
| `gelu_inplace` | gllm-kernels | FFN 激活 | CPU |
| `backend.flash_attention` | Backend | 自注意力 | **可GPU** |
| `rope_apply_inplace` | gllm-kernels | 位置编码 | CPU, 可选 |

### ARCH-INF-003: Generator 推理 - Dense (GeneratorModel)

**调用链**：`forward() → Layers → Attention(w/Cache) + FFN → LmHead` + `sample()`

| 算子 | 来源 | 调用位置 | 说明 |
|------|------|----------|------|
| `linear_forward` | gllm-kernels | Q/K/V/O, FFN, LmHead | CPU |
| `rms_norm_forward` | gllm-kernels | 层归一化 | CPU |
| `rms_norm_inplace` | gllm-kernels | 最终归一化 | CPU |
| `silu_inplace` | gllm-kernels | FFN 激活 (SwiGLU) | CPU |
| `rope_apply_inplace` | gllm-kernels | 位置编码 | CPU |
| `backend.flash_attention` | Backend | 自注意力 + KV Cache | **可GPU** |
| `backend.argmax` | Backend | 贪心采样 | **可GPU** |

### ARCH-INF-004: Generator 推理 - MoE (MoEGeneratorModel)

**调用链**：`forward() → Layers → Attention(w/Cache) + MoE → LmHead` + `sample()`

| 算子 | 来源 | 调用位置 | 说明 |
|------|------|----------|------|
| `linear_forward` | gllm-kernels | Q/K/V/O, Expert FFN, LmHead | CPU |
| `rms_norm_forward` | gllm-kernels | 层归一化 | CPU |
| `rms_norm_inplace` | gllm-kernels | 最终归一化 | CPU |
| `silu_inplace` | gllm-kernels | Expert FFN 激活 | CPU |
| `rope_apply_inplace` | gllm-kernels | 位置编码 | CPU |
| `backend.flash_attention` | Backend | 自注意力 + KV Cache | **可GPU** |
| `backend.moe_route` | Backend | 稀疏专家路由 | **可GPU** |
| `backend.argmax` | Backend | 贪心采样 | **可GPU** |

### ARCH-INF-005: 推理类型算子汇总矩阵

| 算子 | Embeddings | Reranker | Dense Gen | MoE Gen |
|------|:----------:|:--------:|:---------:|:-------:|
| **Backend trait 方法** |
| flash_attention | ✅ | ✅ | ✅ | ✅ |
| paged_attention | - | - | ✅* | ✅* |
| argmax | - | - | ✅ | ✅ |
| moe_route | - | - | - | ✅ |
| **gllm-kernels 函数** |
| linear_forward | ✅ | ✅ | ✅ | ✅ |
| layer_norm_forward | ✅ | ✅ | - | - |
| rms_norm_forward | - | - | ✅ | ✅ |
| rms_norm_inplace | - | - | ✅ | ✅ |
| gelu_inplace | ✅ | ✅ | ✅** | - |
| silu_inplace | - | - | ✅ | ✅ |
| rope_apply_inplace | ✅*** | ✅*** | ✅ | ✅ |

*: 分页注意力为可选优化
**: GELU 仅在 hidden_act="gelu" 时使用，多数模型使用 SiLU
***: RoPE 仅在启用位置编码时使用

---

## 架构优化机会

### ARCH-OPT-001: GPU 加速优化空间

当前仅 2/15 算子有 GPU 优化，以下算子有较大优化空间：

| 优先级 | 算子 | 计算占比 | 当前 | 优化方案 | 预期加速 |
|:------:|------|:--------:|------|----------|:--------:|
| **P0-1** | linear_forward | ~70% | CPU | GPU matmul/cuBLAS | **8-16x** |
| **P0-2** | rms_norm | ~10% | CPU | GPU reduce kernel | 2-5x |
| **P0-3** | silu_inplace | ~5% | CPU | GPU element-wise | 3-5x |
| P1 | argmax | <1% | CPU回退 | GPU reduce | 低 |
| P1 | moe_route | <1% | CPU回退 | GPU routing | 低 |

**关键发现**：
- `linear_forward` 占推理计算量的 **70%**，是最高优先级优化目标
- P0 级别算子组合可实现整体 **5-10x** 加速
- `rms_norm` 和 `silu_inplace` 可与前后算子融合，减少内存带宽

### ARCH-OPT-002: 算法集合无状态化

**问题**：当前 Backend 实现包含 GPU handle 状态，但多数算子是无状态的纯函数。

**分析结果**：

| 组件 | 字段数 | 是否无状态 | 可移除 new() |
|------|:------:|:----------:|:------------:|
| `CpuBackend` | 0 | ✅ 完全无状态 | ✅ 可以 |
| `DynamicPooler` | 2 (enum+flag) | ✅ 配置仅 | ✅ 可以 |
| `WgpuBackend` | 3+ | ❌ GPU 资源 | ❌ 不可 |
| `CudaBackend` | 3+ | ❌ GPU 资源 | ❌ 不可 |
| `GeneratorModel` | 5+ | ❌ 权重+缓存 | ❌ 不可 |
| `MoEGeneratorModel` | 6+ | ❌ 权重+专家 | ❌ 不可 |
| `DynamicBertModel` | 4+ | ❌ 权重 | ❌ 不可 |

**优化方向**：
1. `CpuBackend` 可改为零大小类型（ZST），移除 `new()` 方法
2. `DynamicPooler` 可改为构造时配置，无需 `new()` 方法
3. GPU Backend 必须保留状态（设备句柄、缓冲池）
4. 模型组件必须保留状态（权重、KV Cache）

### ARCH-OPT-003: Backend 静态化（OnceLock 模式）

**问题**：当前 `Arc<dyn Backend>` 在每个组件间传递，增加代码复杂度。

**方案**：使用 `std::sync::OnceLock` 实现全局单例。

```rust
// 静态 Backend 实例
static BACKEND: OnceLock<Arc<dyn Backend>> = OnceLock::new();

pub fn get_backend() -> &'static Arc<dyn Backend> {
    BACKEND.get_or_init(|| auto_select_backend())
}

// 使用时直接调用
let backend = get_backend();
backend.flash_attention(...);
```

**收益分析**：

| 维度 | 当前 | 静态化后 | 改善 |
|------|------|----------|------|
| 参数传递点 | 18 处 | 0 处 | -18 |
| 构造函数参数 | 10 个含 backend | 0 个 | -10 |
| 代码行数 | ~150 行传递代码 | ~20 行初始化 | -130 行 |
| 运行时开销 | 每次 Arc clone | 零（静态引用） | 微优化 |

**影响范围**：

| 文件 | 修改内容 |
|------|----------|
| `gllm-kernels/src/backend.rs` | 添加 OnceLock + get_backend() |
| `gllm/src/engine.rs` | 移除 backend 参数 |
| `gllm/src/generator_engine.rs` | 移除 backend 参数 |
| `gllm/src/generator_model.rs` | 移除 backend 字段/参数 |
| `gllm/src/moe_generator_model.rs` | 移除 backend 字段/参数 |
| `gllm/src/dynamic_bert.rs` | 移除 backend 字段/参数 |
| `gllm/src/causal_attention.rs` | 直接调用 get_backend() |
| `gllm/src/moe_layer.rs` | 直接调用 get_backend() |

**约束**：
- 必须在首次推理前调用 `get_backend()` 完成初始化
- 进程生命周期内 Backend 不可更换（符合当前设计）
- 测试时可通过 `GLLM_BACKEND=cpu` 环境变量强制指定

---

## 待实现架构设计

### ARCH-ADR-010: 模型快速加载（并行解析 + 按需下载）

**关联需求**: REQ-LOAD-001

**问题背景**:
- 多分片 SafeTensors 文件（70B 模型可达 15+ 分片）
- 当前顺序解析：shard-1 解析 → shard-2 解析 → ... 串行执行
- **主要场景是本地已缓存**（~/.gllm/models/），不是下载

**现有缓存机制**:
```
ModelManager::prepare(model_id)
    │
    ├── 检查本地缓存 ~/.gllm/models/{repo_id}/
    │   ├── 存在 + 有权重文件 + 有 config.json → 直接加载（主要场景）
    │   └── 缺失 → 下载后加载（首次使用）
    │
    └── 当前瓶颈：多分片顺序解析，即使本地已缓存也慢
```

**设计目标**:
- 本地缓存并行解析：加载速度提升 ≥ 2x
- 首次下载时流水线：下载与解析并行
- 内存峰值可控
- 兼容现有 API

---

#### 1. 两种场景时序图

**场景 A：本地已缓存（主要场景，占 90%+）**

```
优化前（顺序解析）：
shard-1: [===解析===]
shard-2:              [===解析===]
shard-3:                           [===解析===]
                                                → 完成

优化后（并行解析）：
shard-1: [===解析===]
shard-2: [===解析===]     → 完成（Nx 加速，N=CPU核数）
shard-3: [===解析===]
```

**场景 B：首次下载（占 <10%）**

```
优化后（流水线）：
网络层:  [shard-1下载][shard-2下载][shard-3下载]...
解析层:       [shard-1解析][shard-2解析][shard-3解析]...
                                                    → 完成
```

---

#### 2. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     ModelManager（入口）                         │
├─────────────────────────────────────────────────────────────────┤
│  prepare(model_id)                                              │
│      │                                                          │
│      ├── 检查本地缓存                                            │
│      │   ├── 存在 → ParallelParser 直接解析                     │
│      │   └── 缺失 → Downloader 下载 + ParallelParser 解析       │
│      │                                                          │
│      └── WeightLoader 加载到 WeightMatrix/Vector                │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ ParallelParser  │  │ Downloader      │  │ WeightLoader    │
│ rayon + mmap    │  │ hf-hub（可选）  │  │ 现有，不变      │
│ 并行解析多分片  │  │ 按需下载缺失    │  │ SafeTensors→W*  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

#### 3. 核心数据结构

**加载配置（LoadConfig）**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| max_parse_threads | usize | num_cpus | 最大并行解析线程数 |
| use_mmap | bool | true | 是否使用 mmap（推荐） |
| progress_callback | Option<Fn> | None | 进度回调 |

**加载进度（LoadProgress）**:

| 字段 | 类型 | 说明 |
|------|------|------|
| stage | ProgressStage | 当前阶段 |
| current_shard | usize | 当前分片 |
| total_shards | usize | 总分片数 |
| bytes_loaded | u64 | 已加载字节 |

**进度阶段枚举**:

| 值 | 说明 |
|----|------|
| CheckingCache | 检查本地缓存 |
| Downloading | 下载中（仅首次） |
| Parsing | 并行解析中 |
| LoadingWeights | 加载权重 |
| Complete | 完成 |

---

#### 4. 并行解析策略（核心优化）

**rayon + mmap 并行解析**:

```
ParallelParser::parse_shards(shard_files: Vec<PathBuf>)
    │
    ├── shard_files.par_iter()  // rayon 并行迭代
    │   .map(|path| {
    │       let mmap = Mmap::map(File::open(path));  // 零拷贝映射
    │       let tensors = SafeTensors::deserialize(&mmap);
    │       (path, tensors, mmap)  // 保持 mmap 存活
    │   })
    │   .collect()
    │
    └── 返回 Vec<(PathBuf, SafeTensors, Mmap)>
```

**为什么 mmap 而非 read**:

| 方式 | 内存占用 | 加载速度 | 说明 |
|------|----------|----------|------|
| fs::read() | 完整文件复制到堆 | 慢，需要复制 | 当前方式 |
| mmap | 零拷贝，OS 管理 | 快，按需加载 | 优化方式 |

---

#### 5. 与现有组件集成

**不改动 WeightLoader**：

```
ParallelParser 输出          WeightLoader（现有）
     │                              │
     ▼                              ▼
Vec<(shard_path, SafeTensors)> → from_bytes(&mmap_data) → WeightMatrix/Vector
                                    │
                                    └── 现有逻辑不变，只是并行调用
```

**ModelManager 修改点**:

| 方法 | 修改 |
|------|------|
| prepare() | 调用 ParallelParser 替代顺序解析 |
| download_model() | 保持不变（hf-hub 下载） |

---

#### 6. 错误处理

| 错误 | 触发条件 | 处理 |
|------|----------|------|
| MmapError | 文件映射失败 | 回退到 fs::read() |
| ParseError | SafeTensors 格式错误 | 返回错误 |
| PartialShard | 分片文件不完整 | 删除后重新下载 |

---

#### 7. 性能预期

| 场景 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 本地已缓存 7B（单分片） | ~2s | ~2s | 无变化 |
| 本地已缓存 70B（15分片） | ~30s | ~5-10s | 3-6x |
| 首次下载 70B | ~300s | ~150s | ~2x |

**状态**: ✅ 设计完成

### ARCH-ADR-011: 原生量化推理 Kernel

**关联需求**: REQ-QUANT-001

**问题背景**:
- 当前量化推理流程：Q4 权重 → dequantize → f32 matmul → 输出
- 每次推理都重复 dequantize，计算和内存双重浪费
- Q4_0 模型 dequantize 后内存占用增加 4 倍

**设计目标**:
- 直接在量化权重上计算，避免 dequantize 开销
- 推理速度提升 ≥ 2x
- 内存占用降低（Q4: 75%，Q8: 50%）
- 精度损失 < 1%（对比 dequantize 方案）

---

#### 1. 计算流程对比

```
优化前（dequantize）：
Q4 权重 → [dequantize] → f32 → [matmul] → 输出
          ↑ 每次推理都执行，4x 内存膨胀

优化后（原生 kernel）：
Q4 权重 → [Q4 matmul kernel] → 输出
          ↑ in-kernel dequant，零额外内存
```

---

#### 2. 支持的量化格式

**GGUF 格式**:

| 格式 | block_size | 数据布局 | 精度损失 |
|------|------------|----------|----------|
| Q4_0 | 32 | 16 bytes data + 2 bytes scale | 低 |
| Q4_K | 256 | 分块 scale + min | 极低 |
| Q8_0 | 32 | 32 bytes data + 2 bytes scale | 极低 |

**AWQ 格式**:

| 格式 | group_size | 数据布局 | 精度损失 |
|------|------------|----------|----------|
| AWQ INT4 | 128 | packed int4 + scale + zero | 低 |

---

#### 3. 量化数据结构

**Q4_0 Block（GGUF）**:

| 字段 | 偏移 | 大小 | 类型 | 说明 |
|------|------|------|------|------|
| data | 0 | 16 bytes | u8[16] | 32 个 4-bit 值打包 |
| scale | 16 | 2 bytes | f16 | 缩放因子 |

解码公式：`value[i] = (nibble[i] - 8) * scale`

**Q4_K Block（GGUF）**:

| 字段 | 偏移 | 大小 | 类型 | 说明 |
|------|------|------|------|------|
| scales | 0 | 12 bytes | u8[12] | 子块缩放因子（6-bit 打包） |
| d | 12 | 2 bytes | f16 | 超级块缩放 |
| dmin | 14 | 2 bytes | f16 | 超级块最小值缩放 |
| data | 16 | 128 bytes | u8[128] | 256 个 4-bit 值打包 |

**Q8_0 Block（GGUF）**:

| 字段 | 偏移 | 大小 | 类型 | 说明 |
|------|------|------|------|------|
| data | 0 | 32 bytes | i8[32] | 32 个 8-bit 有符号值 |
| scale | 32 | 2 bytes | f16 | 缩放因子 |

解码公式：`value[i] = data[i] * scale`

**AWQ INT4 Layout**:

| 字段 | 大小 | 说明 |
|------|------|------|
| qweight | N/8 * M bytes | 打包的 4-bit 权重 |
| scales | N/group_size * M * 2 bytes | 每组缩放因子（f16） |
| qzeros | N/group_size * M/8 bytes | 打包的 4-bit 零点 |

解码公式：`value[i] = (qweight[i] - qzeros[group]) * scales[group]`

---

#### 4. Backend trait 扩展

**新增方法签名**:

| 方法 | 输入参数 | 输出 | 说明 |
|------|----------|------|------|
| q4_matmul | input: &[f16], weight: Q4Tensor, output: &mut [f16], M, N, K | Result<()> | Q4_0 矩阵乘 |
| q4k_matmul | input: &[f16], weight: Q4KTensor, output: &mut [f16], M, N, K | Result<()> | Q4_K 矩阵乘 |
| q8_matmul | input: &[f16], weight: Q8Tensor, output: &mut [f16], M, N, K | Result<()> | Q8_0 矩阵乘 |
| awq_matmul | input: &[f16], weight: AwqTensor, output: &mut [f16], M, N, K | Result<()> | AWQ INT4 矩阵乘 |

**参数定义**:

| 参数 | 类型 | 说明 |
|------|------|------|
| input | &[f16] | 输入激活 [M, K] 行优先 |
| weight | Q*Tensor | 量化权重 |
| output | &mut [f16] | 输出 [M, N] 行优先 |
| M | usize | batch * seq_len |
| N | usize | 输出维度（hidden_size） |
| K | usize | 输入维度 |

---

#### 5. 后端实现优先级

遵循 gllm-kernels 后端实现优先级：

| 优先级 | 后端 | 实现策略 |
|--------|------|----------|
| P0 | CPU | 参考实现，逐 block 解码 + FMA |
| P1 | WGPU | WGSL compute shader，跨平台 |
| P2 | CUDA | cuBLAS 无原生支持，自定义 kernel |
| P3 | Metal | Metal Performance Shaders 扩展 |
| P4 | ROCm | HIP kernel，参考 CUDA 实现 |

---

#### 6. WGSL Kernel 设计要点

**Q4_0 MatMul Kernel 架构**:

```
┌─────────────────────────────────────────────────────────┐
│ Workgroup (16x16 threads)                                │
├─────────────────────────────────────────────────────────┤
│ Shared Memory:                                           │
│   - tile_A[16][64]: f16, 从 input 加载                   │
│   - scales[64]: f16, 从 Q4 blocks 加载 scale             │
├─────────────────────────────────────────────────────────┤
│ 计算流程（每 thread 负责 1 个输出元素）：                  │
│   1. 协作加载 input tile 到 shared memory                │
│   2. 协作加载 Q4 scales 到 shared memory                 │
│   3. 逐 block 循环：                                     │
│      a. 加载 Q4 packed data (16 bytes = 32 values)       │
│      b. In-register dequant: nibble → f16               │
│      c. FMA: acc += dequant_val * input_val             │
│   4. 写回 output                                         │
├─────────────────────────────────────────────────────────┤
│ 优化点：                                                 │
│   - 向量化加载（vec4<u32>）                              │
│   - Scale 缓存到 shared memory                          │
│   - 展开内层循环（#pragma unroll）                       │
│   - 使用 fma() 指令                                      │
└─────────────────────────────────────────────────────────┘
```

**Workgroup 配置**:

| 量化格式 | workgroup_size | tile_M | tile_N | tile_K |
|----------|---------------|--------|--------|--------|
| Q4_0 | 256 (16x16) | 16 | 16 | 64 |
| Q4_K | 256 (16x16) | 16 | 16 | 256 |
| Q8_0 | 256 (16x16) | 16 | 16 | 32 |
| AWQ | 256 (16x16) | 16 | 16 | 128 |

---

#### 7. CUDA Kernel 设计要点

**Q4_0 MatMul CUDA Kernel**:

```
┌─────────────────────────────────────────────────────────┐
│ Grid: (N/128, M/128)                                     │
│ Block: (128 threads)                                     │
├─────────────────────────────────────────────────────────┤
│ Shared Memory:                                           │
│   - smem_A[128][32+padding]: f16                        │
│   - smem_scales[32]: f16                                │
│   - smem_qdata[32][16]: u8 (Q4 packed)                  │
├─────────────────────────────────────────────────────────┤
│ 计算流程：                                               │
│   1. 双缓冲预取：                                        │
│      - Buffer 0 计算时，Buffer 1 加载下一 tile           │
│   2. 向量化加载：                                        │
│      - LDG.128 加载 Q4 packed data                       │
│      - __half2 向量化 FMA                                │
│   3. Warp-level 优化：                                   │
│      - __shfl_sync 交换 partial sum                      │
│      - Warp-level reduce                                 │
├─────────────────────────────────────────────────────────┤
│ 优化点：                                                 │
│   - 使用 __half2 进行 2-way SIMD                         │
│   - Shared memory bank conflict 避免（padding）          │
│   - 寄存器分块（register tiling）                        │
│   - 双缓冲隐藏内存延迟                                   │
└─────────────────────────────────────────────────────────┘
```

---

#### 8. gllm 侧接口设计

**NativeQLinear 层**:

| 属性 | 类型 | 说明 |
|------|------|------|
| weight | QTensor | 量化权重（Q4/Q8/AWQ） |
| bias | Option<Vec<f16>> | 可选偏置 |
| in_features | usize | 输入维度 |
| out_features | usize | 输出维度 |
| quant_type | QuantType | 量化类型枚举 |

**QuantType 枚举**:

| 值 | 说明 |
|----|------|
| Q4_0 | GGUF Q4_0 格式 |
| Q4_K | GGUF Q4_K 格式 |
| Q8_0 | GGUF Q8_0 格式 |
| AWQ_INT4 | AWQ INT4 格式 |

**QTensor 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| data | Vec<u8> | 量化数据（原始字节） |
| shape | (usize, usize) | 逻辑形状 (out_features, in_features) |
| quant_type | QuantType | 量化类型 |
| block_size | usize | 块大小（Q4_0=32, Q4_K=256, Q8_0=32） |

**forward 方法**:

| 参数 | 类型 | 说明 |
|------|------|------|
| input | &[f16] | 输入 [batch, seq_len, in_features] |
| 返回 | Result<Vec<f16>> | 输出 [batch, seq_len, out_features] |

---

#### 9. 与现有组件集成

**与 QTensor/AwqWeight 集成**:

```
加载流程：
SafeTensors → QTensor（现有）→ NativeQLinear（新增）
                  │
                  └─→ Backend::q*_matmul（新增）

现有代码路径（保留兼容）：
QTensor → dequantize() → f32 → linear_forward

新代码路径（性能优化）：
QTensor → NativeQLinear::forward → Backend::q*_matmul
```

**检测与回退机制**:

| 条件 | 行为 |
|------|------|
| Backend 支持 q*_matmul | 使用原生 kernel |
| Backend 不支持 | 回退到 dequantize + linear_forward |
| GPU kernel 执行失败 | 静默回退 CPU 原生实现 |

---

#### 10. 精度验证策略

**验证方法**:

| 测试类型 | 验证内容 | 通过标准 |
|----------|----------|----------|
| 单元测试 | 单 block 解码正确性 | bit-exact |
| 矩阵乘测试 | 小矩阵 (128x128) 结果 | 相对误差 < 1e-3 |
| 模型级测试 | 完整推理输出 | perplexity 差异 < 0.1% |
| 回归测试 | 对比 dequantize 方案 | 最大误差 < 1% |

**测试矩阵**:

| 量化格式 | 测试规模 | 比较基准 |
|----------|----------|----------|
| Q4_0 | M=1,32,128, N=4096, K=4096 | dequantize + f32 matmul |
| Q4_K | M=1,32,128, N=4096, K=4096 | dequantize + f32 matmul |
| Q8_0 | M=1,32,128, N=4096, K=4096 | dequantize + f32 matmul |
| AWQ | M=1,32,128, N=4096, K=4096 | dequantize + f32 matmul |

---

#### 11. 性能预期

| 量化格式 | 内存节省 | 带宽节省 | 计算加速 | 精度损失 |
|----------|----------|----------|----------|----------|
| Q4_0 | 75% | 75% | ~2x | < 1% |
| Q4_K | 75% | 75% | ~2x | < 0.5% |
| Q8_0 | 50% | 50% | ~1.5x | < 0.1% |
| AWQ INT4 | 75% | 75% | ~2x | < 1% |

**基准测试场景**:
- 模型：Qwen2.5-7B-Q4_0
- 硬件：RTX 4090 / Apple M2 / AMD RX 7900
- 指标：tokens/sec，内存占用，首 token 延迟

**状态**: ✅ 设计完成

---

## 数据结构审计与重构计划

### ARCH-AUDIT-001: 数据结构与类型审计

**审计日期**: 2025-01-26

**审计范围**: gllm 项目核心数据流中的类型定义、转换开销、内存分配模式

---

#### 1. 类型重复定义问题

| 类型 | 位置 | 存储方式 | 问题 |
|------|------|----------|------|
| `FFNWeights` | decoder_layer.rs:11-17 | `Vec<f32>` (gate/up/down) | 与 LinearWeights 功能重复 |
| `LinearWeights` | weight_loader.rs:191-194 | `WeightMatrix` + `Option<WeightVector>` | 更规范的抽象 |
| `ExpertWeights` | moe_layer.rs:9-13 | `Vec<f32>` (gate/up/down) | 与 FFNWeights 完全相同 |

**问题描述**：同一概念（线性层权重）有多种表示方式，导致：
- 代码重复
- 转换开销（在不同表示间转换）
- 维护困难

---

#### 2. 数据转换开销问题

| 转换 | 位置 | 频率 | 开销 |
|------|------|------|------|
| F16/BF16 → f32 | weight_loader.rs:141-185 `convert_to_f32()` | 每次加载 | **高**：所有权重都转换 |
| Q4/Q8 → f32 | quantized.rs `dequantize()` | 每次推理 | **极高**：每次 matmul 都解量化 |
| AWQ INT4 → f32 | awq.rs:58 `dequantize()` | 每次推理 | **极高**：同上 |
| LoadedTensor → WeightMatrix | weight_loader.rs:57 `.clone()` | 每次加载 | **中**：不必要的复制 |
| LoadedTensor → WeightVector | weight_loader.rs:69,77 `.clone()` | 每次加载 | **中**：同上 |

**数据流图（当前状态）**：

```
SafeTensors (F16/BF16)
    │
    ├─→ convert_to_f32() ──→ Vec<f32> ──→ .clone() ──→ WeightMatrix
    │       [CPU转换]         [分配]        [复制]        [包装]
    │
    └─→ 理想路径：直接 mmap 为 &[f16]，GPU 端保持 F16
```

**量化数据流（当前状态）**：

```
GGUF/AWQ 量化权重
    │
    └─→ QTensor (Vec<u8>)
            │
            └─→ dequantize() ──→ Vec<f32> ──→ linear_forward()
                  [每次推理]      [4x内存]     [计算]
```

---

#### 3. 内存分配模式问题

| 结构 | 位置 | 分配模式 | 问题 |
|------|------|----------|------|
| `KVCache` | kv_cache.rs:9-13 | `Vec<Vec<f32>>` | 每层独立分配，内存碎片化 |
| `forward()` 返回 | decoder_layer.rs:51 | 返回 `Vec<f32>` | 每次调用分配新内存 |
| `ffn_forward()` | decoder_layer.rs:81-131 | 分配 gate, up, output | 三个临时 Vec |
| `forward_with_cache()` | causal_attention.rs:154-217 | 分配 q,k,v,attn_out,output | 五个临时 Vec |
| `repeat_kv()` | causal_attention.rs:232-256 | 即使 repeat=1 也分配 | 不必要的复制 |

**热路径内存分配统计（单次 forward）**：

| 位置 | 分配次数 | 大小 | 可避免 |
|------|----------|------|--------|
| DecoderLayer.forward | 1 | seq_len * hidden_size | ✅ 可复用 |
| ffn_forward | 3 | seq_len * intermediate_size × 2 + seq_len * hidden_size | ✅ 可复用 |
| forward_with_cache | 5 | q + k + v + attn_out + output | ✅ 可复用 |
| repeat_kv | 2 | 当 GQA repeat > 1 时必要 | 部分可避免 |

---

#### 4. 不必要的 clone/to_vec 调用

| 位置 | 代码 | 问题 | 修复建议 |
|------|------|------|----------|
| causal_attention.rs:235 | `return (k.to_vec(), v.to_vec())` | repeat=1 时不必要复制 | 返回 Cow<[f32]> |
| decoder_layer.rs:61-65 | `.iter().zip().map().collect()` | 创建新 Vec | 原地操作 |
| generator_model.rs:111 | `logits.to_vec()` | 用于温度缩放 | 原地操作 |
| moe_generator_model.rs:160 | `logits.to_vec()` | 同上 | 原地操作 |
| hooks.rs:299 | `hidden_states.to_vec()` | Hook 捕获 | 可选深拷贝 |

---

#### 5. 类型不一致问题

| 数据 | 存储格式 | 计算格式 | gllm-kernels 期望 |
|------|----------|----------|-------------------|
| 模型权重 | SafeTensors (F16/BF16) | Vec<f32> | &[f32] 或 WeightMatrix |
| KV Cache | Vec<f32> | &[f32] | TensorSlice::F32 |
| 量化权重 | Vec<u8> (packed) | Vec<f32> (dequantized) | 原生 Q*Tensor |
| 激活值 | Vec<f32> | &[f32] | TensorSlice |

**理想状态**：
- 权重保持 F16 存储，GPU 端计算转换
- 量化权重保持 packed 格式，in-kernel dequantize
- 激活值使用预分配缓冲区

---

### ARCH-REFACTOR-001: 重构工作计划

**目标**：消除不必要的类型转换和内存分配，提升推理性能

---

#### 阶段 1：类型统一（低风险）

| 任务 | 优先级 | 影响范围 | 预期收益 |
|------|--------|----------|----------|
| 1.1 统一 FFNWeights 和 LinearWeights | P1 | decoder_layer, moe_layer | 代码简化 |
| 1.2 移除 LoadedTensor.clone() | P0 | weight_loader.rs | 减少加载时内存占用 50% |
| 1.3 使用 Cow<[f32]> 替代 to_vec() | P1 | causal_attention, decoder_layer | 减少不必要复制 |

**1.1 详细设计**：

```
现状：
├─ FFNWeights { gate_proj: Vec<f32>, ... }
└─ LinearWeights { weight: WeightMatrix, ... }

目标：
└─ 统一使用 LinearWeights，FFNWeights 改为：
   struct FFN {
       gate: LinearWeights,
       up: LinearWeights,
       down: LinearWeights,
   }
```

**1.2 详细设计**：

```
现状：
LoadedTensor::to_weight_matrix(&self) → WeightMatrix::new(self.data.clone(), ...)

目标：
LoadedTensor::into_weight_matrix(self) → WeightMatrix::new(self.data, ...)
                                         ^^ 消费 self，避免 clone
```

---

#### 阶段 2：内存复用（中风险）

| 任务 | 优先级 | 影响范围 | 预期收益 |
|------|--------|----------|----------|
| 2.1 引入 ScratchBuffer 工作区 | P0 | decoder_layer, causal_attention | 减少 80% 临时分配 |
| 2.2 KVCache 连续内存布局 | P1 | kv_cache.rs | 减少碎片，提升缓存命中 |
| 2.3 forward() 接受输出缓冲区 | P1 | 所有 forward 方法 | 允许外部管理内存 |

**2.1 ScratchBuffer 设计**：

```
struct ScratchBuffer {
    // 预分配的工作区，按最大需求分配
    ffn_gate: Vec<f32>,      // max_batch * max_seq * intermediate_size
    ffn_up: Vec<f32>,        // 同上
    ffn_output: Vec<f32>,    // max_batch * max_seq * hidden_size
    attn_q: Vec<f32>,        // max_batch * max_seq * num_heads * head_dim
    attn_k: Vec<f32>,        // 同上（KV heads）
    attn_v: Vec<f32>,        // 同上
    attn_out: Vec<f32>,      // 同上
}

impl ScratchBuffer {
    fn new(config: &ModelConfig, max_batch: usize, max_seq: usize) -> Self;
    fn get_ffn_workspace(&mut self, batch: usize, seq: usize) -> FFNWorkspace;
    fn get_attn_workspace(&mut self, batch: usize, seq: usize) -> AttnWorkspace;
}
```

**2.2 KVCache 连续布局设计**：

```
现状：
KVCache {
    k_cache: Vec<Vec<f32>>,  // [num_layers][max_seq * kv_heads * head_dim]
    v_cache: Vec<Vec<f32>>,  // 每层独立分配
}

目标：
KVCache {
    // 单次分配，连续内存
    data: Vec<f32>,  // [2 * num_layers * max_seq * kv_heads * head_dim]

    fn k_layer(&self, layer: usize) -> &[f32];  // 视图访问
    fn v_layer(&self, layer: usize) -> &[f32];
}
```

---

#### 阶段 3：零拷贝加载（高收益）

| 任务 | 优先级 | 影响范围 | 预期收益 |
|------|--------|----------|----------|
| 3.1 F16 直通加载 | P0 | weight_loader | 消除 F16→f32 转换 |
| 3.2 mmap 权重访问 | P1 | weight_loader, safetensors | 零内存复制加载 |
| 3.3 GPU 端 F16 计算 | P2 | Backend trait | 进一步减少转换 |

**3.1 F16 直通设计**：

```
现状：
SafeTensors (F16) → convert_to_f32() → Vec<f32> → WeightMatrix

目标（保持 API 兼容）：
SafeTensors (F16) → WeightMatrixF16 → Backend::linear_f16()
                                      ↓
                                 （仅在 CPU 回退时转换为 f32）
```

**3.2 mmap 设计**：

```
struct MmapWeight<'a> {
    data: &'a [u8],      // mmap 直接引用
    shape: (usize, usize),
    dtype: Dtype,
}

impl MmapWeight {
    fn as_f16_slice(&self) -> &[f16];  // 零拷贝视图
    fn to_f32_vec(&self) -> Vec<f32>;  // 按需转换（回退路径）
}
```

---

#### 阶段 4：量化直通（依赖 ARCH-ADR-011）

| 任务 | 优先级 | 依赖 | 预期收益 |
|------|--------|------|----------|
| 4.1 消除 dequantize 开销 | P0 | ARCH-ADR-011 实现 | 4x 内存节省 + 2x 加速 |
| 4.2 原生量化 KV Cache | P2 | 4.1 完成 | 进一步压缩 Cache |

---

#### 实施优先级矩阵

| 优先级 | 任务 | 风险 | 收益 | 依赖 |
|--------|------|------|------|------|
| **P0** | 1.2 移除 clone() | 低 | 中 | 无 |
| **P0** | 2.1 ScratchBuffer | 中 | 高 | 无 |
| **P0** | 3.1 F16 直通 | 中 | 高 | 无 |
| P1 | 1.1 类型统一 | 低 | 低 | 无 |
| P1 | 1.3 Cow 替代 to_vec | 低 | 低 | 无 |
| P1 | 2.2 KVCache 连续化 | 中 | 中 | 无 |
| P1 | 2.3 输出缓冲区参数 | 中 | 中 | 2.1 |
| P1 | 3.2 mmap 权重 | 高 | 高 | 3.1 |
| P2 | 3.3 GPU F16 计算 | 高 | 高 | 3.1 + Backend 支持 |
| P2 | 4.1 量化直通 | 高 | 极高 | ARCH-ADR-011 |

---

#### 验收标准

| 阶段 | 指标 | 基准 | 目标 |
|------|------|------|------|
| 阶段 1 | 加载时内存峰值 | 100% | ≤ 50% |
| 阶段 2 | 推理时临时分配次数 | ~10 次/forward | ≤ 2 次/forward |
| 阶段 3 | 模型加载时间 (7B) | 基准 | ≤ 50% |
| 阶段 4 | 量化模型推理速度 | dequant 方案 | ≥ 2x |

**状态**: ✅ 审计完成，待实施

---

### ARCH-AUDIT-002: 代码级架构违规审计

**审计日期**: 2025-01-26

**审计范围**: gllm 和 gllm-kernels 项目中违反架构原则的代码模式

---

#### 1. 内存-GPU 数据移动违规（12处）

**原则**: CPU 计算全在内存，GPU 计算全在显存，数据传输仅在推理开始/结束时发生

| 位置 | 文件:行号 | 问题描述 | 优先级 |
|------|-----------|----------|--------|
| WGPU 类型转换 | wgpu_backend.rs:142-144 | 热路径 T→f32 转换创建临时 Vec | P0 |
| Staging Buffer | wgpu_backend.rs:231-283 | readback_f32() 每次创建 staging buffer | P0 |
| RmsNorm | wgpu_backend.rs:314-315 | GPU buffer 创建+readback 每次调用 | P0 |
| Softmax | wgpu_backend.rs:349 | 同上 | P1 |
| Linear | wgpu_backend.rs:388-389 | 同上 | P0 |
| Attention | wgpu_backend.rs:446 | 同上 | P0 |
| Apply Temp | wgpu_backend.rs:498 | 同上 | P1 |
| Paged Attn | paged_attn/dispatch.rs:130-164 | 每次 attention 创建 7 个 GPU buffers | **P0** |
| repeat_kv | causal_attention.rs:232-255 | repeat=1 时仍执行复制 | P1 |
| Layer 循环 | generator_model.rs:70-81 | 每层创建新 hidden Vec（80层=80次分配） | P0 |
| FFN 分配 | decoder_layer.rs:81-131 | 热路径 3 次 Vec 分配 | P0 |
| Residual | decoder_layer.rs:61-65 | 手写 add 创建新 Vec | P1 |

**修复方案**:
- **GPU Buffer Pool**: 在 WgpuBackend 引入缓冲池，避免每次调用创建/销毁
- **Staging Buffer 复用**: 维护可复用的 staging buffer 池
- **ScratchBuffer**: 为热路径预分配工作区（见 ARCH-REFACTOR-001 §2.1）

---

#### 2. 后端选择违规（4处）

**原则**: 后端选择仅在程序启动时执行一次（OnceLock 模式）

| 位置 | 文件:行号 | 问题描述 | 影响 |
|------|-----------|----------|------|
| build_embedding_backend | engine.rs:505-510 | 重复调用 detect_backend() | 冗余检测 |
| build_rerank_backend | engine.rs:523-530 | 同上 | 同上 |
| build_generator_backend | engine.rs:590-599 | 同上 | 同上 |
| build_backend | engine.rs:626-635 | 同上 | 同上 |

**根因分析**:
- `Engine::new()` 调用 `auto_select_backend()`
- 各 `build_*_backend()` 方法再次调用 `detect_backend()`
- 同一 Engine 实例内重复检测后端

**修复方案**:
- 实现 ARCH-OPT-003 Backend 静态化（OnceLock 模式）
- 移除 `build_*_backend()` 中的 `detect_backend()` 调用
- 所有后端获取通过 `get_backend()` 静态方法

---

#### 3. 不必要的数据复制（12处）

**原则**: 零拷贝/零分配，避免不必要的 Vec 分配、.clone()、.to_vec()

| 位置 | 文件:行号 | 问题描述 | 数据量 | 优先级 |
|------|-----------|----------|--------|--------|
| repeat_kv | causal_attention.rs:235 | repeat=1 时 .to_vec() | ~MB | P1 |
| sample | generator_model.rs:111 | temperature=1.0 时 logits.to_vec() | ~KB | P2 |
| 权重加载 | weight_loader.rs:57 | .clone() 创建 WeightMatrix | **GB级** | **P0** |
| 权重加载 | weight_loader.rs:69 | .clone() 创建 bias | MB级 | P0 |
| 权重加载 | weight_loader.rs:77 | .clone() 创建 norm | MB级 | P0 |
| KV Cache | paged_attention.rs:80 | get_kv() 克隆整个 cache | **MB级/每次** | **P0** |
| MoE token | moe_layer.rs:89 | 循环内 Vec 分配 | KB×tokens | P1 |
| MoE gate | moe_layer.rs:114 | expert_forward 内分配 | KB×experts | P1 |
| MoE up | moe_layer.rs:126 | 同上 | 同上 | P1 |
| MoE down | moe_layer.rs:146 | 同上 | 同上 | P1 |
| Pooling | engine.rs:325 | pooled_rows 每行 .to_vec() | KB×rows | P2 |
| MoE sample | moe_generator_model.rs:160 | 同 generator_model.rs:111 | ~KB | P2 |

**修复方案**:
- **所有权转移**: `into_weight_matrix(self)` 替代 `.clone()`
- **Cow<[f32]>**: 条件复制，仅必要时分配
- **原地操作**: 使用 gllm-kernels 的 `*_inplace` 算子
- **预分配工作区**: ScratchBuffer 避免循环内分配

---

#### 4. 算子使用不正确（11处）

**原则**: 使用 gllm-kernels 向量化算子，禁止手写标量循环

| 位置 | 文件:行号 | 问题代码 | 正确算子 | 优先级 |
|------|-----------|----------|----------|--------|
| Residual add | decoder_layer.rs:61-65 | `.iter().zip().map(a+b).collect()` | `add_inplace()` | P0 |
| SwiGLU | decoder_layer.rs:113-116 | 分开的 silu + mul | `silu_mul_inplace()` | P0 |
| ReLU | dynamic_bert.rs:508-514 | `if x < 0 { 0 } else { x }` 循环 | `relu_inplace()` | P1 |
| Tanh | dynamic_bert.rs:518-522 | `x.tanh()` 循环 | `tanh_inplace()` | P1 |
| Sigmoid | dynamic_bert.rs:524-526 | `1.0 / (1.0 + (-x).exp())` 循环 | `sigmoid_scalar()` | P1 |
| Pos Embed | dynamic_bert.rs:571-588 | 3 层嵌套循环加位置嵌入 | `add_inplace()` | P0 |
| Pos Embed | dynamic_bert.rs:590-610 | 同上（另一个 case） | `add_inplace()` | P0 |

**修复方案**:
- 替换手写循环为 gllm-kernels 算子调用
- 考虑算子融合（如 residual_add_rms_norm_fused）
- 确保 SIMD 向量化（已在 gllm-kernels CPU 后端实现）

---

#### 5. 违规统计与优先级汇总

| 类别 | 违规数 | P0 | P1 | P2 | 性能影响 |
|------|--------|----|----|----|---------|
| 内存-GPU 数据移动 | 12 | 6 | 5 | 1 | **极高** |
| 后端选择 | 4 | 4 | 0 | 0 | 中 |
| 不必要数据复制 | 12 | 4 | 5 | 3 | **高** |
| 算子使用不正确 | 11 | 4 | 3 | 4 | 高 |
| **总计** | **39** | **18** | **13** | **8** | - |

**预期收益**:

| 修复类别 | 预期加速 | 内存节省 |
|----------|----------|----------|
| GPU Buffer Pool | 2-5x（WGPU 路径） | 减少 GPU 内存碎片 |
| 后端单例化 | 微优化 | 代码简化 |
| 零拷贝加载 | 1.5-2x 加载速度 | **50% 峰值内存** |
| 算子替换 | 1.5-3x（CPU 路径） | - |

**状态**: ✅ 审计完成

---

### ARCH-REFACTOR-002: 代码级重构计划

**基于**: ARCH-AUDIT-002 发现

---

#### 阶段 0：GPU 缓冲管理（最高优先级）

| 任务 | 优先级 | 影响文件 | 预期收益 |
|------|--------|----------|----------|
| 0.1 WgpuBackend 引入 BufferPool | **P0** | wgpu_backend.rs | 2-5x GPU 路径加速 |
| 0.2 Staging Buffer 复用 | **P0** | wgpu_backend.rs | 减少 GPU 同步开销 |
| 0.3 Paged Attention 缓冲池 | **P0** | paged_attn/dispatch.rs | 每次 attn 减少 7 次分配 |

**0.1 BufferPool 设计**:

```
struct BufferPool {
    // 按大小分桶的可复用 buffer
    buckets: HashMap<usize, Vec<wgpu::Buffer>>,

    fn acquire(&mut self, size: usize) -> wgpu::Buffer;  // 获取或创建
    fn release(&mut self, buffer: wgpu::Buffer);         // 归还复用
}

使用模式：
let buffer = pool.acquire(size);
// 使用 buffer...
pool.release(buffer);  // 不销毁，归还池中
```

**0.2 Staging Buffer 复用设计**:

```
struct StagingBufferManager {
    read_staging: Option<wgpu::Buffer>,   // 复用的读取 staging
    write_staging: Option<wgpu::Buffer>,  // 复用的写入 staging
    max_size: usize,                      // 当前最大容量

    fn get_read_staging(&mut self, size: usize) -> &wgpu::Buffer;
    fn get_write_staging(&mut self, size: usize) -> &wgpu::Buffer;
}
```

---

#### 阶段 1：后端单例化

| 任务 | 优先级 | 影响文件 | 预期收益 |
|------|--------|----------|----------|
| 1.1 实现 ARCH-OPT-003 OnceLock | **P0** | backend.rs, engine.rs | 代码简化 |
| 1.2 移除冗余 detect_backend() | P0 | engine.rs | 消除启动时冗余检测 |

---

#### 阶段 2：零拷贝权重加载

| 任务 | 优先级 | 影响文件 | 预期收益 |
|------|--------|----------|----------|
| 2.1 into_weight_matrix() | **P0** | weight_loader.rs | 50% 加载内存节省 |
| 2.2 KV Cache 引用返回 | **P0** | paged_attention.rs | 消除每次 get 的 MB 级复制 |
| 2.3 Cow<[f32]> 条件复制 | P1 | causal_attention.rs | 减少不必要复制 |

**2.2 KV Cache 引用设计**:

```
现状（违规）：
fn get_kv(&self) -> (Vec<f32>, Vec<f32>) {
    (keys.clone(), values.clone())  // MB 级复制
}

目标：
fn get_kv(&self) -> (&[f32], &[f32]) {  // 零拷贝引用
    (&self.keys, &self.values)
}

// 需要修改时才复制
fn get_kv_mut(&mut self) -> (&mut [f32], &mut [f32]);
```

---

#### 阶段 3：算子替换

| 任务 | 优先级 | 影响文件 | 正确算子 |
|------|--------|----------|----------|
| 3.1 Residual add 替换 | **P0** | decoder_layer.rs | `add_inplace()` |
| 3.2 SwiGLU 融合 | **P0** | decoder_layer.rs | `silu_mul_inplace()` |
| 3.3 位置嵌入向量化 | **P0** | dynamic_bert.rs | `add_inplace()` |
| 3.4 激活函数替换 | P1 | dynamic_bert.rs | `relu/tanh/sigmoid_inplace()` |

---

#### 阶段 4：热路径内存复用

| 任务 | 优先级 | 影响文件 | 预期收益 |
|------|--------|----------|----------|
| 4.1 ScratchBuffer（见 ARCH-REFACTOR-001 §2.1） | **P0** | 全局 | 减少 80% 临时分配 |
| 4.2 MoE 循环优化 | P1 | moe_layer.rs | 减少循环内分配 |
| 4.3 Generator hidden 复用 | P0 | generator_model.rs | 消除每层分配 |

**4.3 Generator hidden 复用设计**:

```
现状（违规）：
for layer in &self.layers {
    let new_hidden = layer.forward(&hidden, ...);  // 每层分配
    hidden = new_hidden;
}

目标：
let mut hidden_a = vec![0.0; size];  // 预分配双缓冲
let mut hidden_b = vec![0.0; size];
for (i, layer) in self.layers.iter().enumerate() {
    let (input, output) = if i % 2 == 0 {
        (&hidden_a, &mut hidden_b)
    } else {
        (&hidden_b, &mut hidden_a)
    };
    layer.forward_into(input, output, ...);  // 原地写入
}
```

---

#### 实施优先级矩阵（更新）

| 优先级 | 任务 | 风险 | 收益 | 依赖 |
|--------|------|------|------|------|
| **P0** | 0.1 WgpuBackend BufferPool | 中 | **极高** | 无 |
| **P0** | 0.2 Staging Buffer 复用 | 低 | 高 | 无 |
| **P0** | 0.3 Paged Attention 缓冲池 | 中 | **极高** | 0.1 |
| **P0** | 1.1 OnceLock 后端单例 | 低 | 中 | 无 |
| **P0** | 2.1 into_weight_matrix | 低 | 高 | 无 |
| **P0** | 2.2 KV Cache 引用返回 | 中 | **高** | 无 |
| **P0** | 3.1 Residual add 替换 | 低 | 中 | 无 |
| **P0** | 3.2 SwiGLU 融合 | 低 | 中 | 无 |
| **P0** | 4.3 Generator hidden 复用 | 中 | 高 | 无 |
| P1 | 1.2 移除冗余 detect | 低 | 低 | 1.1 |
| P1 | 2.3 Cow 条件复制 | 低 | 低 | 无 |
| P1 | 3.3 位置嵌入向量化 | 低 | 中 | 无 |
| P1 | 3.4 激活函数替换 | 低 | 低 | 无 |
| P1 | 4.2 MoE 循环优化 | 中 | 中 | 4.1 |

---

#### 验收标准（更新）

| 阶段 | 指标 | 基准 | 目标 |
|------|------|------|------|
| 阶段 0 | WGPU 单次 forward GPU 分配 | ~10 次 | ≤ 2 次 |
| 阶段 1 | 后端检测调用次数 | ~4 次/Engine | 1 次/进程 |
| 阶段 2 | 模型加载峰值内存 | 100% | ≤ 50% |
| 阶段 3 | CPU 算子耗时（BERT forward） | 基准 | ≤ 50% |
| 阶段 4 | 推理时临时分配次数 | ~10 次/forward | ≤ 2 次/forward |

**状态**: ✅ 审计完成，待实施

---

### ARCH-ROADMAP-001: 完整优化路线图（CPU/GPU 分类）

> **目的**：汇总所有审计发现和待实施优化任务，明确标注每个任务的后端类型

#### 后端类型说明

| 标记 | 含义 | 说明 |
|------|------|------|
| **[CPU]** | CPU 后端 | 纯 Rust 实现，无 GPU 依赖 |
| **[WGPU]** | WGPU 后端 | 跨平台 GPU（Vulkan/DX12/Metal） |
| **[CUDA]** | CUDA 后端 | NVIDIA GPU 专用 |
| **[Metal]** | Metal 后端 | Apple Silicon 专用 |
| **[ROCm]** | ROCm 后端 | AMD GPU 专用 |
| **[全后端]** | 所有后端 | 架构层变更，影响所有后端 |

---

#### 一、P0-GPU: GPU 核心算子加速（gllm-kernels）

> **来源**：ARCH-OPT-001 GPU 加速优化空间
> **状态**: ✅ 已完成 (2026-01-26) - 所有核心算子已有 GPU 实现

| ID | 任务 | 后端 | 影响 | 预期收益 | 状态 |
|----|------|------|------|----------|------|
| P0-GPU-1 | linear_forward GPU 实现 | **[WGPU]** **[CUDA]** **[Metal]** **[ROCm]** | 计算量 70% | 8-16x 加速 | ✅ |
| P0-GPU-2 | rms_norm GPU 实现 | **[WGPU]** **[CUDA]** **[Metal]** **[ROCm]** | 每层 2 次调用 | 2-5x 加速 | ✅ |
| P0-GPU-3 | silu_inplace GPU 实现 | **[WGPU]** **[CUDA]** **[Metal]** **[ROCm]** | element-wise | 3-5x 加速 | ✅ |

---

#### 二、P0-MEM: 内存与 Buffer 管理优化

> **来源**：ARCH-AUDIT-002 内存-GPU 数据移动违规（12 处）
> **状态**: ✅ 全部完成 (2026-01-26) [commit: 10cb10a, c34ff82]

| ID | 任务 | 后端 | 位置 | 问题 | 状态 |
|----|------|------|------|------|------|
| P0-MEM-1 | BufferPool 单例实现 | **[WGPU]** | wgpu_backend.rs | 每次 forward 重新分配 | ✅ |
| P0-MEM-2 | StagingBuffer 复用 | **[WGPU]** | wgpu_backend.rs | 每次 map_read 重新创建 | ✅ |
| P0-MEM-3 | PagedAttention dispatch 优化 | **[WGPU]** | paged_attn/dispatch.rs | 热路径 Vec 分配 | ✅ |
| P0-MEM-4 | KV Cache buffer 预分配 | **[CPU]** | generator_model.rs:445-467 | 逐 token 扩展 | ✅ 已实现 |
| P0-MEM-5 | decoder attention 输出复用 | **[CPU]** | decoder_layer.rs:178-195 | 重复 .to_vec() | ✅ |
| P0-MEM-6 | WeightLoader 零拷贝 | **[CPU]** | weight_loader.rs:89-156 | SafeTensor 多次克隆 | ✅ |
| P0-MEM-7 | PagedAttention 索引预分配 | **[CPU]** | paged_attention.rs:234-278 | 循环内 Vec 扩展 | ✅ |
| P0-MEM-8 | MoE routing 批量计算 | **[CPU]** | moe_layer.rs:123-189 | 逐 token .to_vec() | ✅ |

---

#### 三、P0-ARCH: 架构层优化

> **来源**：ARCH-AUDIT-002 后端选择违规（4 处）
> **状态**: ✅ 已完成 (2026-01-26) [commit: 10cb10a]

| ID | 任务 | 后端 | 位置 | 问题 | 状态 |
|----|------|------|------|------|------|
| P0-ARCH-1 | OnceLock 单次后端检测 | **[全后端]** | engine.rs:505-635 | 每次初始化重复检测 | ✅ |
| P0-ARCH-2 | BackendType 静态确定 | **[全后端]** | engine.rs | 运行时重复判断 | ✅ |

---

#### 四、P0-OPS: 算子实现替换

> **来源**：ARCH-AUDIT-002 未正确使用 gllm-kernels 算子（11 处）
> **状态**: ✅ 已验证 (2026-01-26) - 代码已使用 gllm_kernels 算子

| ID | 任务 | 后端 | 位置 | 当前实现 | 替换为 | 状态 |
|----|------|------|------|----------|--------|------|
| P0-OPS-1 | softmax 替换 | **[CPU]** | decoder_layer.rs:234-256 | 手写循环 | gllm_kernels::softmax | ✅ 已使用 |
| P0-OPS-2 | layer_norm 替换 | **[CPU]** | dynamic_bert.rs:178-201 | 手写循环 | gllm_kernels::layer_norm | ✅ 已使用 |
| P0-OPS-3 | gelu 替换 | **[CPU]** | dynamic_bert.rs:156-167 | 手写 tanh 近似 | gllm_kernels::gelu | ✅ 已使用 |
| P0-OPS-4 | rope_embedding 替换 | **[CPU]** | decoder_layer.rs:289-334 | 手写三角函数 | gllm_kernels::rope | ✅ 已使用 |
| P0-OPS-5 | attention_scores 替换 | **[CPU]** | decoder_layer.rs:178-195 | 手写 matmul+scale | gllm_kernels::attention | ✅ FlashAttn |
| P0-OPS-6 | cross_entropy 替换 | **[CPU]** | generator_model.rs:234-256 | 手写循环 | gllm_kernels::cross_entropy | N/A 无此场景 |

---

#### 五、P1-LOAD: 模型加载优化

> **来源**：ARCH-ADR-010 异步并行模型加载
> **状态**: ✅ 已完成 (2026-01-26) [commit: cd9338f] REQ-LOAD-001

| ID | 任务 | 后端 | 说明 | 状态 |
|----|------|------|------|------|
| P1-LOAD-1 | AsyncShardLoader 实现 | **[CPU]** | 多分片并行下载 | ✅ parallel_parser.rs |
| P1-LOAD-2 | MmapWeightLoader 实现 | **[CPU]** | 内存映射权重加载 | ✅ ShardBytes::Mmap |
| P1-LOAD-3 | 进度回调接口 | **[CPU]** | LoadProgress trait | ✅ LoadProgress |

---

#### 六、P2-QUANT: 原生量化推理

> **来源**：ARCH-ADR-011 原生量化推理 Kernel
> **状态**: ✅ CPU 参考实现已完成 (2026-01-26) [commit: 66af66e, e94a31ee] REQ-QUANT-001

| ID | 任务 | 后端 | 说明 | 状态 |
|----|------|------|------|------|
| P2-QUANT-1 | Q4_0 数据结构 | **[CPU]** | Block 定义 | ✅ gllm-kernels |
| P2-QUANT-2 | Q4_0 WGSL kernel | **[WGPU]** | in-kernel dequant | 🔲 |
| P2-QUANT-3 | Q4_0 CUDA kernel | **[CUDA]** | shared memory 优化 | 🔲 |
| P2-QUANT-4 | AWQ 数据结构 | **[CPU]** | AwqPackedWeight | ✅ gllm-kernels |
| P2-QUANT-5 | AWQ WGSL kernel | **[WGPU]** | 分组反量化 | 🔲 |
| P2-QUANT-6 | AWQ CUDA kernel | **[CUDA]** | tensor core 利用 | 🔲 |

---

#### 七、P1-MISC: 其他优化

> **来源**：ARCH-AUDIT-001 数据结构审计、ARCH-OPT-001 无状态算法

| ID | 任务 | 后端 | 说明 |
|----|------|------|------|
| P1-MISC-1 | TokenizerWrapper 简化 | **[CPU]** | 移除冗余字段 |
| P1-MISC-2 | ScratchBuffer 实现 | **[CPU]** | 热路径预分配工作区 |
| P1-MISC-3 | PromptCache 实现 | **[CPU]** | 相同 prompt 复用 |
| P1-MISC-4 | 算子纯函数化 | **[CPU]** | 无状态算法集合 |
| P1-MISC-5 | KV 压缩策略 | **[CPU]** | 长序列内存优化 |

---

#### 任务统计

| 后端类型 | 任务数 | 占比 |
|----------|--------|------|
| **[CPU]** | 20 | 60.6% |
| **[WGPU]** | 6 | 18.2% |
| **[CUDA]** | 3 | 9.1% |
| **[Metal]** | 3 | 9.1% |
| **[ROCm]** | 3 | 9.1% |
| **[全后端]** | 2 | 6.1% |
| **总计** | 33 | - |

> 注：P0-GPU 任务需要在 WGPU/CUDA/Metal/ROCm 四个后端分别实现，因此每个任务计为 4 个后端

---

#### 执行路线图

```
Phase 1: P0 核心优化（建议顺序）
├─ P0-GPU-1 linear_forward → 最大收益（70%计算量）
├─ P0-MEM-1/2 BufferPool → 减少 GPU 分配开销
├─ P0-ARCH-1/2 OnceLock → 架构基础
└─ P0-OPS-1~6 算子替换 → CPU 性能提升

Phase 2: P1 增强优化
├─ P1-LOAD-1~3 → 大模型加载体验
├─ P1-MISC-1~5 → 内存和缓存优化
└─ P0-GPU-2/3 → 补充 GPU 算子

Phase 3: P2 高级特性
└─ P2-QUANT-1~6 → 量化推理支持
```

---

#### 验收标准汇总

| 指标 | 基准 | 目标 | 来源 |
|------|------|------|------|
| WGPU forward GPU 分配 | ~10 次 | ≤ 2 次 | P0-MEM |
| 后端检测调用 | ~4 次/Engine | 1 次/进程 | P0-ARCH |
| 模型加载峰值内存 | 100% | ≤ 50% | P1-LOAD |
| CPU 算子耗时 | 基准 | ≤ 50% | P0-OPS |
| linear_forward 延迟 | CPU 基准 | ≥ 8x 加速 | P0-GPU-1 |
| 量化模型内存 | FP16 基准 | Q4: 25%, Q8: 50% | P2-QUANT |

**状态**: ⚠️ 路线图需更新（发现 P0-CRITICAL 级别问题）

---

### ARCH-AUDIT-003: CPU/GPU 内存架构审计（关键缺陷）

> **严重程度**：🔴 CRITICAL - 当前架构导致 50-100x 性能损失
>
> **根本问题**：数据始终在 CPU 内存，GPU 沦为"远程协处理器"

#### 问题概述

| 指标 | 当前状态 | 理想状态 | 差距 |
|------|----------|----------|------|
| 单 token 延迟 | 5-10 秒 | 50-100ms | **50-100x** |
| GPU 显存利用率 | <10% | >80% | **8x** |
| PCIe 传输/token | ~256MB | ~0 | ∞ |
| 生成 100 tokens | 8-12 分钟 | 5-10 秒 | **60-100x** |

#### 数据流对比

**当前流程（错误）**：
```
┌─────────────────────────────────────────────────────────────┐
│                      CPU 内存 (RAM)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ 权重      │  │ KV Cache │  │ 激活值    │  │ 中间结果  │    │
│  │ Vec<f32> │  │ Vec<f32> │  │ Vec<f32> │  │ Vec<f32> │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │           │
│       ▼             ▼             ▼             ▼           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PCIe 总线 (瓶颈: 32GB/s)                │   │
│  └─────────────────────────────────────────────────────┘   │
│       │             │             │             │           │
└───────┼─────────────┼─────────────┼─────────────┼───────────┘
        ▼             ▼             ▼             ▼
┌───────────────────────────────────────────────────────────┐
│                      GPU 显存 (VRAM)                       │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ 临时 Buffer（每次算子调用重新分配，用完释放）          │ │
│  │ 仅用于 Flash Attention 等单个算子                     │ │
│  └──────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘

每生成 1 个 token：
  - 上传权重: ~14GB (7B 模型)
  - 上传 KV Cache: ~256MB (ctx=2048)
  - 上传激活: ~数 MB
  - 下载结果: ~数 MB
  - PCIe 往返: ~14GB × 2 = 28GB
  - 传输时间: 28GB / 32GB/s = ~0.9 秒 (仅传输!)
```

**理想流程（GPU 驻留）**：
```
┌─────────────────────────────────────────────────────────────┐
│                      CPU 内存 (RAM)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 仅存放：输入 token IDs、输出 logits、配置             │   │
│  └──────────────────────────────────────────────────────┘   │
│       │ (一次性加载)      ▲ (仅下载 logits, KB 级)         │
└───────┼───────────────────┼─────────────────────────────────┘
        ▼                   │
┌───────────────────────────────────────────────────────────┐
│                      GPU 显存 (VRAM)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 权重      │  │ KV Cache │  │ 激活值    │  │ 中间结果  │  │
│  │ GpuBuffer│  │ GpuBuffer│  │ GpuBuffer│  │ GpuBuffer│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                           │
│  所有计算在 GPU 上完成，零 PCIe 往返                       │
│  - Embedding lookup (GPU)                                 │
│  - Linear Q/K/V (GPU)                                     │
│  - RoPE (GPU)                                             │
│  - Flash Attention (GPU)                                  │
│  - FFN (GPU)                                              │
│  - LayerNorm (GPU)                                        │
└───────────────────────────────────────────────────────────┘

每生成 1 个 token：
  - PCIe 传输: ~0 (权重已在 GPU)
  - GPU 计算: ~50-100ms
  - 下载 logits: ~128KB (vocab_size × 4 bytes)
```

---

#### 问题清单（按严重程度排序）

| ID | 问题 | 位置 | 当前实现 | 影响 |
|----|------|------|----------|------|
| **CRIT-1** | **KV Cache 在 CPU** | kv_cache.rs:8-19 | `Vec<Vec<f32>>` | **50x 性能损失** |
| **CRIT-2** | **权重在 CPU** | weights.rs:17-28 | `Vec<f32>` | **10x 性能损失** |
| **CRIT-3** | **线性投影在 CPU** | causal_attention.rs:219-228 | CPU matmul | **5x 性能损失** |
| HIGH-1 | 中间激活在 CPU | causal_attention.rs:167-174 | `Vec<f32>` | 3x 性能损失 |
| HIGH-2 | RoPE 在 CPU | causal_attention.rs:177-178 | CPU 计算 | 2x 性能损失 |
| HIGH-3 | repeat_kv 在 CPU | causal_attention.rs:232-255 | CPU 循环 | 2x 性能损失 |
| MED-1 | 无 Buffer 抽象 | backend_trait.rs | 无统一接口 | 架构问题 |
| MED-2 | 无计算图融合 | 全链路 | 逐算子执行 | 5x 性能损失 |

---

#### ARCH-BUFFER-001: GPU Buffer 抽象层设计

##### 核心概念

```
┌─────────────────────────────────────────────────────────┐
│                    TensorBuffer Trait                    │
├─────────────────────────────────────────────────────────┤
│ + location() -> MemoryLocation                          │
│ + shape() -> &[usize]                                   │
│ + dtype() -> DType                                      │
│ + as_cpu_slice() -> Option<&[T]>                        │
│ + as_gpu_ptr() -> Option<GpuPtr>                        │
│ + to_device(device: Device) -> Result<Self>             │
│ + download() -> Result<CpuBuffer>                       │
│ + upload_from(data: &[T]) -> Result<()>                 │
└─────────────────────────────────────────────────────────┘
           ▲                              ▲
           │                              │
┌──────────┴──────────┐      ┌───────────┴───────────┐
│     CpuBuffer       │      │      GpuBuffer        │
├─────────────────────┤      ├─────────────────────────┤
│ data: Vec<T>        │      │ ptr: *mut c_void        │
│ shape: Vec<usize>   │      │ size: usize             │
│                     │      │ device: Device          │
│                     │      │ backend: BackendType    │
└─────────────────────┘      └─────────────────────────┘
                                       ▲
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
           ┌────────┴───────┐ ┌────────┴───────┐ ┌───────┴────────┐
           │  WgpuBuffer    │ │  CudaBuffer    │ │  MetalBuffer   │
           │  (wgpu::Buffer)│ │  (CUdeviceptr) │ │  (MTLBuffer)   │
           └────────────────┘ └────────────────┘ └────────────────┘
```

##### 内存位置枚举

| 位置 | 说明 | 传输成本 |
|------|------|----------|
| `MemoryLocation::Cpu` | CPU 主内存 | - |
| `MemoryLocation::Gpu(device_id)` | GPU 显存 | CPU↔GPU 需要 PCIe |
| `MemoryLocation::Unified` | 统一内存（支持的平台） | 自动迁移 |
| `MemoryLocation::Pinned` | 锁页内存 | 快速 DMA |

##### 权重容器升级

| 字段 | 当前 | 升级后 |
|------|------|--------|
| WeightMatrix.data | `Vec<f32>` | `Arc<dyn TensorBuffer>` |
| WeightVector.data | `Vec<f32>` | `Arc<dyn TensorBuffer>` |
| KVCache.k_cache | `Vec<Vec<f32>>` | `Vec<GpuBuffer>` |
| KVCache.v_cache | `Vec<Vec<f32>>` | `Vec<GpuBuffer>` |

##### 关键接口

| 接口 | 说明 |
|------|------|
| `GpuBuffer::alloc(size, device)` | 在 GPU 上分配显存 |
| `GpuBuffer::from_cpu(data)` | CPU→GPU 上传 |
| `GpuBuffer::to_cpu()` | GPU→CPU 下载 |
| `GpuBuffer::copy_from(other)` | GPU 内拷贝（零 PCIe） |
| `BufferPool::get(size)` | 从池中获取预分配 buffer |
| `BufferPool::release(buffer)` | 归还到池中 |

---

#### ARCH-RESIDENT-001: GPU 驻留推理设计

##### 初始化阶段（一次性）

| 步骤 | 操作 | 数据位置 |
|------|------|----------|
| 1 | 加载 SafeTensors 到 CPU | CPU |
| 2 | 上传权重到 GPU | CPU → **GPU** |
| 3 | 释放 CPU 权重副本 | CPU 释放 |
| 4 | 预分配 KV Cache buffers | **GPU** |
| 5 | 预分配激活 buffers | **GPU** |

##### 推理阶段（每 token）

| 步骤 | 操作 | 数据位置 | PCIe 传输 |
|------|------|----------|-----------|
| 1 | 上传 token IDs | CPU → GPU | ~4 bytes |
| 2 | Embedding lookup | GPU | 0 |
| 3 | 32×Decoder Layer | GPU | 0 |
| 4 | LM Head | GPU | 0 |
| 5 | 下载 logits | GPU → CPU | ~128KB |

**总 PCIe 传输**: ~128KB/token (对比当前 ~28GB/token)

##### Decoder Layer 内部（全 GPU）

```
输入 (GpuBuffer)
    │
    ├─→ RMSNorm (GPU) ─→ Q/K/V Linear (GPU) ─→ RoPE (GPU)
    │                                              │
    │   ┌──────────────────────────────────────────┘
    │   ▼
    │   Flash Attention (GPU) ← KV Cache (GpuBuffer, 原地更新)
    │   │
    │   ▼
    │   Output Linear (GPU)
    │   │
    ├───┴─→ Residual Add (GPU)
    │
    ├─→ RMSNorm (GPU) ─→ Gate/Up Linear (GPU) ─→ SiLU (GPU)
    │                                              │
    │                                              ▼
    │                                         Down Linear (GPU)
    │                                              │
    └──────────────────────────────────────────────┴─→ Residual Add (GPU)
                                                              │
                                                              ▼
                                                         输出 (GpuBuffer)
```

---

#### 更新后的优化路线图

##### P0-CRITICAL: GPU 驻留优化（最高优先级）

| ID | 任务 | 后端 | 预期收益 | 复杂度 |
|----|------|------|----------|--------|
| **P0-CRIT-1** | **TensorBuffer 抽象层** | **[全后端]** | 架构基础 | 高 |
| **P0-CRIT-2** | **权重 GPU 驻留** | **[WGPU][CUDA][Metal][ROCm]** | **10x** | 中 |
| **P0-CRIT-3** | **KV Cache GPU 驻留** | **[WGPU][CUDA][Metal][ROCm]** | **50x** | 中 |
| **P0-CRIT-4** | **激活值 GPU 驻留** | **[WGPU][CUDA][Metal][ROCm]** | **5x** | 中 |
| **P0-CRIT-5** | **全链路 GPU 推理** | **[WGPU][CUDA][Metal][ROCm]** | **3x** | 高 |

##### P0-CRITICAL 依赖关系

```
P0-CRIT-1 (Buffer 抽象)
    │
    ├─→ P0-CRIT-2 (权重 GPU)
    │       │
    │       └─→ P0-CRIT-5 (全链路)
    │               ▲
    ├─→ P0-CRIT-3 (KV Cache GPU) ─┘
    │
    └─→ P0-CRIT-4 (激活 GPU) ────────┘
```

##### 完整优先级矩阵（更新）

| 优先级 | 类别 | 任务数 | 预期总收益 |
|--------|------|--------|------------|
| **P0-CRITICAL** | GPU 驻留 | 5 | **50-100x** |
| P0-GPU | GPU 算子 | 3 | 8-16x |
| P0-MEM | 内存管理 | 8 | 2-5x |
| P0-ARCH | 架构优化 | 2 | 2x |
| P0-OPS | 算子替换 | 6 | 2x |
| P1-LOAD | 模型加载 | 3 | 2x |
| P2-QUANT | 量化推理 | 6 | 4x |
| P1-MISC | 其他 | 5 | 1.5x |

---

#### 验收标准（P0-CRITICAL）

| 指标 | 当前 | 目标 | 验证方法 |
|------|------|------|----------|
| 单 token 延迟 | 5-10s | ≤100ms | benchmark |
| GPU 显存利用率 | <10% | >80% | nvidia-smi |
| PCIe 传输/token | ~28GB | <1MB | profiler |
| 权重位置 | CPU | GPU | 代码检查 |
| KV Cache 位置 | CPU | GPU | 代码检查 |
| 推理全链路 | CPU+GPU | 全 GPU | profiler |

**状态**: 🔴 待实施（最高优先级）

---

### ARCH-EXECUTION-PLAN-001: 零妥协极致性能执行计划

> **铁律**：
> - 42 个任务全部必须完成，无一例外
> - 所有后端（CPU/WGPU/CUDA/Metal/ROCm）同步实现，不存在"优先某后端"
> - 所有验收标准必须 100% 达成，不接受"基本达到"
> - 依赖关系是唯一的串行约束，无依赖即并行

---

#### 任务清单（全部强制）

| 类别 | 任务数 | 验收标准 |
|------|--------|----------|
| CRITICAL | 5 | 单 token ≤50ms，GPU 显存利用率 >90% |
| GPU-OPS | 3×5后端=15 | 全后端实现，性能达到理论峰值 80% |
| MEM | 8 | forward 内存分配 = 0（全预分配） |
| ARCH | 2 | 后端检测 = 1次/进程，零运行时开销 |
| OPS | 6 | 全部替换为 gllm-kernels，零手写循环 |
| REFACTOR | 4 | 零 `dyn Trait`，全 enum 静态派发 |
| LOAD | 3 | 70B 模型加载 ≤30s（NVMe SSD） |
| QUANT | 6 | Q4 精度损失 <0.5%，速度 ≥FP16 |
| MISC | 5 | 全部实现，无遗留 |
| **总计** | **42** | **全部 PASS** |

---

#### 依赖图（唯一约束）

```
                                    ┌──────────────────┐
                                    │    STAGE 0       │
                                    │   基础架构层      │
                                    └────────┬─────────┘
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         │                                   │                                   │
         ▼                                   ▼                                   ▼
┌─────────────────┐               ┌─────────────────┐               ┌─────────────────┐
│   REFACTOR-4    │               │   ARCH-1 + 2    │               │   无依赖任务     │
│ BackendImpl     │               │   OnceLock      │               │ REFACTOR-1,2,3  │
│ [gllm-kernels]  │               │   BackendType   │               │ MISC-1,2,3,4    │
└────────┬────────┘               └─────────────────┘               └─────────────────┘
         │
         │ 完成后解锁
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    STAGE 1                                          │
│                              GPU 基础设施层                                          │
│                                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │ CRIT-1      │  │ GPU-1       │  │ GPU-2       │  │ GPU-3       │               │
│  │ TensorBuffer│  │ linear      │  │ rms_norm    │  │ silu        │               │
│  │             │  │ ×5后端      │  │ ×5后端      │  │ ×5后端      │               │
│  └──────┬──────┘  └─────────────┘  └─────────────┘  └─────────────┘               │
│         │                                                                          │
│         │ 完成后解锁                                                               │
│         ▼                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                                │
│  │ CRIT-2      │  │ CRIT-3      │  │ CRIT-4      │  ← 全部并行                     │
│  │ 权重GPU驻留 │  │ KV Cache    │  │ 激活值      │                                │
│  │ ×5后端      │  │ GPU驻留     │  │ GPU驻留     │                                │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                                │
│         │                │                │                                        │
│         └────────────────┼────────────────┘                                        │
│                          ▼                                                          │
│                   ┌─────────────┐                                                   │
│                   │ CRIT-5      │                                                   │
│                   │ 全链路GPU   │                                                   │
│                   └─────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             │ 完成后解锁
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    STAGE 2                                          │
│                              极致优化层（全并行）                                     │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ MEM-1~8      │ OPS-1~6      │ LOAD-1→2→3   │ QUANT-1,4→2,3,5,6 │ MISC-5   │   │
│  │ 内存优化     │ 算子替换     │ 加载优化     │ 量化推理           │ KV压缩   │   │
│  │ (MEM-3依赖   │ 全并行       │ 串行依赖     │ 数据结构→kernel   │          │   │
│  │  MEM-1,2)    │              │              │                   │          │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

#### 详细任务表（按依赖排序）

##### STAGE 0: 基础架构（解锁全部后续）

| ID | 任务 | 项目 | 依赖 | 后端 | 验收标准 |
|----|------|------|------|------|----------|
| **REFACTOR-4** | BackendImpl enum | gllm-kernels | 无 | 全后端 | `Arc<dyn Backend>` 全部替换为 `BackendImpl` |
| **ARCH-1** | OnceLock 后端检测 | gllm | 无 | 全后端 | `auto_select_backend()` 调用 = 1次/进程 |
| **ARCH-2** | BackendType 静态 | gllm | ARCH-1 | 全后端 | 零运行时类型判断 |
| REFACTOR-1 | Trait 重命名 | gllm | 无 | - | 消除 GeneratorModelTrait 歧义 |
| REFACTOR-2 | EmbeddingModel enum | gllm | 无 | - | 零 `Box<dyn>` |
| REFACTOR-3 | GeneratorModel enum | gllm | 无 | - | 零 `Box<dyn>` |
| MISC-1 | TokenizerWrapper | gllm | 无 | - | 移除冗余字段 |
| MISC-2 | ScratchBuffer | gllm | 无 | - | 热路径零分配 |
| MISC-3 | PromptCache | gllm | 无 | - | 相同 prompt 复用 |
| MISC-4 | 算子纯函数化 | gllm | 无 | - | 无状态算法集合 |

**并行组**：REFACTOR-4, ARCH-1, REFACTOR-1/2/3, MISC-1/2/3/4 全部并行启动

---

##### STAGE 1: GPU 基础设施

| ID | 任务 | 项目 | 依赖 | 后端 | 验收标准 |
|----|------|------|------|------|----------|
| **CRIT-1** | TensorBuffer 抽象 | gllm | REFACTOR-4 | 全后端 | 统一 CPU/GPU buffer 接口 |
| **GPU-1** | linear_forward | gllm-kernels | REFACTOR-4 | CPU/WGPU/CUDA/Metal/ROCm | 理论峰值 80%，全后端 |
| **GPU-2** | rms_norm | gllm-kernels | REFACTOR-4 | CPU/WGPU/CUDA/Metal/ROCm | reduce kernel，全后端 |
| **GPU-3** | silu_inplace | gllm-kernels | REFACTOR-4 | CPU/WGPU/CUDA/Metal/ROCm | element-wise，全后端 |
| **CRIT-2** | 权重 GPU 驻留 | gllm | CRIT-1 | WGPU/CUDA/Metal/ROCm | 权重常驻 GPU，零 PCIe 传输 |
| **CRIT-3** | KV Cache GPU | gllm | CRIT-1 | WGPU/CUDA/Metal/ROCm | KV 常驻 GPU，原地更新 |
| **CRIT-4** | 激活值 GPU | gllm | CRIT-1 | WGPU/CUDA/Metal/ROCm | 中间激活全 GPU |
| **CRIT-5** | 全链路 GPU | gllm | CRIT-2,3,4 + GPU-1,2,3 | 全后端 | embed→decode→sample 全 GPU |

**并行组**：
- CRIT-1, GPU-1, GPU-2, GPU-3 并行
- CRIT-2, CRIT-3, CRIT-4 并行（等待 CRIT-1）
- CRIT-5 等待全部完成

---

##### STAGE 2: 极致优化（全并行）

| ID | 任务 | 项目 | 依赖 | 验收标准 |
|----|------|------|------|----------|
| MEM-1 | BufferPool | gllm-kernels | STAGE 1 | WGPU buffer 复用 |
| MEM-2 | StagingBuffer | gllm-kernels | STAGE 1 | map_read 零分配 |
| MEM-3 | PagedAttn dispatch | gllm-kernels | MEM-1,2 | 热路径零 Vec |
| MEM-4 | KV Cache 预分配 | gllm | STAGE 1 | 固定容量，零扩展 |
| MEM-5 | attention 输出复用 | gllm | STAGE 1 | 消除 .to_vec() |
| MEM-6 | WeightLoader 零拷贝 | gllm | STAGE 1 | mmap 直接使用 |
| MEM-7 | PagedAttn 索引 | gllm | STAGE 1 | 预分配索引数组 |
| MEM-8 | MoE routing 批量 | gllm | STAGE 1 | 批量处理 tokens |
| OPS-1 | softmax 替换 | gllm | STAGE 1 | gllm_kernels::softmax |
| OPS-2 | layer_norm 替换 | gllm | STAGE 1 | gllm_kernels::layer_norm |
| OPS-3 | gelu 替换 | gllm | STAGE 1 | gllm_kernels::gelu |
| OPS-4 | rope 替换 | gllm | STAGE 1 | gllm_kernels::rope |
| OPS-5 | attention 替换 | gllm | STAGE 1 | gllm_kernels::attention |
| OPS-6 | cross_entropy 替换 | gllm | STAGE 1 | gllm_kernels::cross_entropy |
| LOAD-1 | AsyncShardLoader | gllm | STAGE 1 | 并行下载 |
| LOAD-2 | MmapWeightLoader | gllm | LOAD-1 | 并行 mmap |
| LOAD-3 | 进度回调 | gllm | LOAD-2 | LoadProgress trait |
| QUANT-1 | Q4_0 结构 | gllm-kernels | STAGE 1 | Block 定义 |
| QUANT-4 | AWQ 结构 | gllm-kernels | STAGE 1 | AwqPackedWeight |
| QUANT-2 | Q4_0 WGSL | gllm-kernels | QUANT-1 | in-kernel dequant |
| QUANT-3 | Q4_0 CUDA | gllm-kernels | QUANT-1 | shared memory |
| QUANT-5 | AWQ WGSL | gllm-kernels | QUANT-4 | 分组反量化 |
| QUANT-6 | AWQ CUDA | gllm-kernels | QUANT-4 | tensor core |
| MISC-5 | KV 压缩 | gllm | CRIT-3 | 长序列优化 |

---

#### 验收标准（全部强制）

| 指标 | 要求 | 验证方法 | 通过条件 |
|------|------|----------|----------|
| **单 token 延迟** | ≤50ms | benchmark | 100次平均 |
| **GPU 显存利用率** | >90% | nvidia-smi | 持续监控 |
| **PCIe 传输/token** | <1KB | profiler | 推理期间 |
| **forward 内存分配** | =0 | valgrind | 无 malloc |
| **后端检测** | =1次/进程 | 日志 | 启动后零调用 |
| **dyn Trait** | =0 | grep | 无匹配 |
| **手写算子** | =0 | grep | 全替换 |
| **70B 加载时间** | ≤30s | benchmark | NVMe SSD |
| **Q4 精度损失** | <0.5% | eval | 对比 FP16 |
| **全后端支持** | 5/5 | CI | 全绿 |

---

#### 执行策略

```
启动时刻 T=0:
├─ 并行启动 STAGE 0 全部 10 个无依赖任务
│   ├─ gllm-kernels: REFACTOR-4
│   └─ gllm: ARCH-1, REFACTOR-1/2/3, MISC-1/2/3/4
│
├─ REFACTOR-4 完成 → 解锁:
│   ├─ CRIT-1 (TensorBuffer)
│   └─ GPU-1/2/3 (全后端并行)
│
├─ CRIT-1 完成 → 解锁:
│   └─ CRIT-2/3/4 (并行)
│
├─ CRIT-2/3/4 + GPU-1/2/3 完成 → 解锁:
│   └─ CRIT-5 (全链路)
│
└─ STAGE 1 完成 → 解锁 STAGE 2 全部任务并行执行
```

**状态**: 🔴 待执行（零妥协方案）

---

### ARCH-REFACTOR-003: 冗余抽象层审计与精简

> **目标**：优雅的、不多一分不少一点的架构和完美的分层设计
>
> **原则**：每一层抽象必须有明确的价值，纯透传或仅2个实现的 trait 应考虑消除

---

#### 问题1：`GeneratorModelTrait` 重复定义（🔴 严重）

**现状**：同名 trait 在两个文件中定义，方法签名完全不同，造成混乱

| 文件 | 方法 | 用途 |
|------|------|------|
| `generator_engine.rs:11-22` | generate, load_safetensors, load_awq, load_gguf, max_position_embeddings | 引擎层加载/生成接口 |
| `generator_model.rs:165-172` | forward, sample, hidden_size, num_layers, vocab_size, max_position_embeddings | 模型层推理接口 |

**问题**：
- 同名不同义，违反最小惊讶原则
- 代码阅读者难以区分
- 存在命名空间污染风险

**建议**：
- 方案A：重命名为 `GeneratorEngineModel` + `GeneratorInferModel`
- 方案B：合并为单一 trait，engine 层直接使用具体类型

---

#### 问题2：`EmbeddingModelTrait` 纯透传（🟡 中等）

**现状**：`engine.rs:316-329`

```rust
pub(crate) trait EmbeddingModelTrait {
    fn forward(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>>;
    fn hidden_size(&self) -> usize;
}

// 实现1：纯透传
impl EmbeddingModelTrait for DynamicBertModel {
    fn forward(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        let hidden_states = DynamicBertModel::forward(self, tokens)?;  // 直接调用同名方法
        let pooled = DynamicBertModel::pool_hidden_states(self, &hidden_states, tokens);
        Ok(pooled_rows(pooled))
    }
    fn hidden_size(&self) -> usize { DynamicBertModel::hidden_size(self) }  // 纯透传
}

// 实现2：纯透传
impl EmbeddingModelTrait for DecoderModel {
    fn forward(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        DecoderModel::forward(self, tokens)  // 纯透传
    }
    fn hidden_size(&self) -> usize { DecoderModel::hidden_size(self) }  // 纯透传
}
```

**问题**：
- 仅 2 个实现，使用 `Box<dyn EmbeddingModelTrait>` 引入动态派发开销
- 实现代码几乎是纯透传，无附加逻辑（除 pool_hidden_states）
- trait 本身没有提供抽象价值

**建议**：使用 enum 替代 trait object

```rust
// 优化后（零成本抽象）
pub(crate) enum EmbeddingModel {
    Bert(DynamicBertModel),
    Decoder(DecoderModel),
}

impl EmbeddingModel {
    #[inline]
    fn forward(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Bert(m) => { /* ... */ }
            Self::Decoder(m) => m.forward(tokens),
        }
    }
}
```

**收益**：消除动态派发，启用内联优化

---

#### 问题3：`Box<dyn GeneratorModelTrait>` 仅2个实现（🟡 中等）

**现状**：`generator_engine.rs:79`

```rust
pub(crate) struct GeneratorEngine {
    model: Box<dyn GeneratorModelTrait>,  // 动态派发
}

// 创建时（line 102-109）
let mut model: Box<dyn GeneratorModelTrait> = match info.architecture {
    Architecture::GLM4MoE | Architecture::Qwen3MoE | ...
        => Box::new(MoEGeneratorModel::new(config, backend)?),
    _ => Box::new(GeneratorModel::new(config, backend)?),
};
```

**问题**：
- 仅 2 个实现：`GeneratorModel`、`MoEGeneratorModel`
- 编译期已知所有可能类型
- 动态派发无必要

**建议**：使用 enum 替代

```rust
pub(crate) enum GeneratorModelImpl {
    Standard(GeneratorModel),
    MoE(MoEGeneratorModel),
}
```

---

#### 问题4：`Arc<dyn Backend>` 动态派发（🔴 需要静态化）

**现状**：12 处使用 `Arc<dyn Backend>`

| 文件 | 行号 | 用途 |
|------|------|------|
| generator_model.rs | 21, 25 | 模型持有 backend |
| moe_generator_model.rs | 68, 72 | MoE 模型持有 |
| decoder_model.rs | 34 | Decoder 模型持有 |
| dynamic_bert.rs | 42, 123, 158, 379 | BERT 模型持有 |
| moe_layer.rs | 33, 42 | MoE 层持有 |
| causal_attention.rs | 112, 120 | 注意力层持有 |

**当前行为**：
- 后端实例在 `auto_select_backend()` 时创建一次
- 之后所有算子调用复用同一个 `Arc<dyn Backend>` 实例
- 不存在重复初始化问题

**架构问题**：
- `dyn Trait` 引入 vtable 间接跳转（非性能瓶颈，但不够优雅）
- 阻止编译器跨函数内联优化
- 与 Rust "零成本抽象"哲学不符

**目标**：改用 enum 实现零成本静态派发，保持运行时后端选择能力

---

##### ARCH-STATIC-BACKEND-001: Backend 静态化设计

**方案**：使用 enum 替代 trait object

```rust
// gllm-kernels/src/backend.rs

/// 零成本 Backend 抽象（编译期静态派发）
#[derive(Clone)]
pub enum BackendImpl {
    Cpu(CpuBackend),
    #[cfg(feature = "wgpu")]
    Wgpu(Arc<WgpuBackend>),  // Arc 用于共享 GPU 资源
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaBackend>),
    #[cfg(feature = "metal")]
    Metal(Arc<MetalBackend>),
    #[cfg(feature = "rocm")]
    Rocm(Arc<RocmBackend>),
}

impl BackendImpl {
    /// 运行时自动选择最佳后端
    pub fn auto_select() -> Self {
        #[cfg(feature = "cuda")]
        if cuda_available() { return Self::Cuda(Arc::new(CudaBackend::new())); }

        #[cfg(feature = "rocm")]
        if rocm_available() { return Self::Rocm(Arc::new(RocmBackend::new())); }

        #[cfg(feature = "metal")]
        if metal_available() { return Self::Metal(Arc::new(MetalBackend::new())); }

        #[cfg(feature = "wgpu")]
        if wgpu_available() { return Self::Wgpu(Arc::new(WgpuBackend::new())); }

        Self::Cpu(CpuBackend::new())
    }

    /// 内联派发（零成本）
    #[inline]
    pub fn matmul(&self, a: TensorSlice, b: TensorSlice, c: TensorSliceMut, config: MatmulConfig) -> Result<(), String> {
        match self {
            Self::Cpu(b) => b.matmul(a, b, c, config),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(b) => b.matmul(a, b, c, config),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.matmul(a, b, c, config),
            #[cfg(feature = "metal")]
            Self::Metal(b) => b.matmul(a, b, c, config),
            #[cfg(feature = "rocm")]
            Self::Rocm(b) => b.matmul(a, b, c, config),
        }
    }

    // ... 其他 17 个方法同理（可用宏生成）
}
```

**宏简化实现**：

```rust
macro_rules! dispatch_backend {
    ($self:expr, $method:ident, $($arg:expr),*) => {
        match $self {
            BackendImpl::Cpu(b) => b.$method($($arg),*),
            #[cfg(feature = "wgpu")]
            BackendImpl::Wgpu(b) => b.$method($($arg),*),
            #[cfg(feature = "cuda")]
            BackendImpl::Cuda(b) => b.$method($($arg),*),
            #[cfg(feature = "metal")]
            BackendImpl::Metal(b) => b.$method($($arg),*),
            #[cfg(feature = "rocm")]
            BackendImpl::Rocm(b) => b.$method($($arg),*),
        }
    };
}

impl BackendImpl {
    #[inline]
    pub fn matmul(&self, a: TensorSlice, b: TensorSlice, c: TensorSliceMut, config: MatmulConfig) -> Result<(), String> {
        dispatch_backend!(self, matmul, a, b, c, config)
    }

    #[inline]
    pub fn flash_attention(&self, q: TensorSlice, k: TensorSlice, v: TensorSlice,
                           output: TensorSliceMut, config: FlashAttentionConfig) -> Result<(), String> {
        dispatch_backend!(self, flash_attention, q, k, v, output, config)
    }

    // ... 宏自动生成其余方法
}
```

**gllm 侧变更**：

```rust
// 之前
pub struct GeneratorModel {
    backend: Arc<dyn Backend>,  // 动态派发
}

// 之后
pub struct GeneratorModel {
    backend: BackendImpl,  // 静态派发，可内联
}
```

**收益**：

| 方面 | 说明 |
|------|------|
| 零成本抽象 | 符合 Rust 设计哲学，enum match 编译为跳转表 |
| 内联优化 | 编译器可跨函数内联，LTO 时优化更充分 |
| 代码一致性 | 全项目统一使用 enum 抽象，无 `dyn Trait` |
| 运行时选择 | 保持 `auto_select()` 运行时检测后端能力 |

> 注：vtable 开销本身可忽略（~ns 级），静态化主要目的是架构纯净性

**Backend trait 保留**：
- trait 定义保留，用于约束各后端实现（`impl Backend for CpuBackend`）
- 消费端从 `Arc<dyn Backend>` 改为 `BackendImpl` enum
- trait 的 18 个方法保持不变，不拆分

---

#### 问题5：FallbackEmbedder 包装器（🟢 合理）

**现状**：`fallback.rs` - 436 行

**分析**：
- 提供 GPU→CPU 自动回退功能
- 包含失败计数、后端切换逻辑
- 是合理的功能封装，非冗余抽象

**结论**：保留，这是有价值的功能层

---

#### 重构优先级矩阵

| ID | 问题 | 优先级 | 风险 | 收益 | 建议 |
|----|------|--------|------|------|------|
| REFACTOR-1 | GeneratorModelTrait 重复 | **P0** | 低 | 高（清晰度） | 重命名或合并 |
| REFACTOR-2 | EmbeddingModelTrait 透传 | **P0** | 低 | 中（性能） | 改用 enum |
| REFACTOR-3 | GeneratorEngine Box<dyn> | **P0** | 低 | 中（性能） | 改用 enum |
| **REFACTOR-4** | **Arc<dyn Backend> 静态化** | **P0** | 中 | **高（零成本）** | **改用 BackendImpl enum** |

> 注：Backend trait 保持 18 个方法不变，不需要拆分

---

#### 理想架构分层（目标状态）

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户 API 层                               │
│  Client, AsyncClient, EmbedderHandle, RerankerHandle            │
│  职责：公开接口、异步包装、连接池                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        引擎层 (Engine)                           │
│  EmbeddingEngine, GeneratorEngine                                │
│  职责：模型生命周期、批处理调度、内存管理                          │
│                                                                 │
│  ❌ 移除：EmbeddingModelTrait（改用 enum EmbeddingModel）         │
│  ❌ 移除：GeneratorModelTrait@engine（改用 enum GeneratorModel）  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        模型层 (Model)                            │
│  DynamicBertModel, GeneratorModel, MoEGeneratorModel             │
│  职责：模型结构定义、前向传播、权重管理                            │
│                                                                 │
│  ✅ 保留：GeneratorInferTrait（统一推理接口，重命名后）            │
│  ✅ 使用：BackendImpl（零成本静态派发）                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        算子层 (Operators) - gllm-kernels         │
│                                                                 │
│  ✅ 保留：Backend trait（18个方法，约束各后端实现）               │
│  ✅ 新增：BackendImpl enum（零成本静态派发）                      │
│                                                                 │
│  pub enum BackendImpl {                                         │
│      Cpu(CpuBackend),                                           │
│      Wgpu(Arc<WgpuBackend>),                                    │
│      Cuda(Arc<CudaBackend>),                                    │
│      Metal(Arc<MetalBackend>),                                  │
│      Rocm(Arc<RocmBackend>),                                    │
│  }                                                              │
│                                                                 │
│  职责：算子定义、静态派发、编译期优化                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        后端层 (Backend Impl)                     │
│  CpuBackend, WgpuBackend, CudaBackend, MetalBackend, RocmBackend │
│  职责：硬件抽象、Kernel 实现、impl Backend for XxxBackend        │
└─────────────────────────────────────────────────────────────────┘
```

**零成本抽象原则**：

| 层级 | 抽象方式 | 派发开销 |
|------|----------|----------|
| 引擎层 | enum（EmbeddingModel, GeneratorModel） | 0（match 编译为跳转表） |
| 模型层 | 具体类型（DynamicBertModel 等） | 0（无抽象） |
| 算子层 | enum BackendImpl | 0（match + 内联） |
| 后端层 | 具体类型（CpuBackend 等） | 0（无抽象） |

**全栈零动态派发**：从用户 API 到 Kernel 调用，无任何 `dyn Trait`

---

#### 代码变更预估

| 组件 | 当前 | 变更后 | 说明 |
|------|------|--------|------|
| **gllm-kernels** | | | |
| backend.rs | 21 | ~150 | 新增 BackendImpl enum + dispatch 宏 |
| **gllm** | | | |
| generator_engine.rs | 137 | ~80 | 移除 trait，改用 enum |
| engine.rs | ~100 | ~60 | 移除 EmbeddingModelTrait |
| 各模型文件 | Arc<dyn Backend> | BackendImpl | 类型替换（12处） |

**净效果**：
- gllm-kernels 增加 ~130 行（BackendImpl 实现）
- gllm 减少 ~100 行（移除冗余 trait）
- **运行时零成本**：消除所有动态派发

---

#### 实施检查清单

| 步骤 | 内容 | 影响范围 | 状态 |
|------|------|----------|------|
| **1** | **gllm-kernels: 新增 BackendImpl enum** | backend.rs | 🔲 |
| **2** | **gllm-kernels: 实现 dispatch_backend! 宏** | backend.rs | 🔲 |
| **3** | **gllm-kernels: BackendImpl 实现所有 18 个方法** | backend.rs | 🔲 |
| **4** | **gllm-kernels: auto_select() 返回 BackendImpl** | backend.rs | 🔲 |
| 5 | gllm: `Arc<dyn Backend>` → `BackendImpl`（12处） | 全模型层 | 🔲 |
| 6 | 重命名 `generator_model.rs::GeneratorModelTrait` → `GeneratorInferTrait` | generator_model.rs | 🔲 |
| 7 | 删除 `generator_engine.rs::GeneratorModelTrait`，改用 enum | generator_engine.rs | 🔲 |
| 8 | `EmbeddingModelTrait` → `enum EmbeddingModel` | engine.rs | 🔲 |
| 9 | `Box<dyn GeneratorModelTrait>` → `enum GeneratorModelImpl` | generator_engine.rs | 🔲 |
| 10 | 更新所有调用点 | 全项目 | 🔲 |
| 11 | 基准测试验证性能提升 | benchmark | 🔲 |

**依赖关系**：
```
步骤 1-4（gllm-kernels）
    ↓
步骤 5（gllm 适配 BackendImpl）
    ↓
步骤 6-9（gllm 内部重构）
    ↓
步骤 10-11（收尾验证）
```

**状态**: ✅ 审计完成，待实施
