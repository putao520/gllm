# gllm 架构设计

## 定位

**gllm = 推理客户端库**

负责模型管理、配置、分词、权重加载，调用 gllm-kernels 后端执行计算。

---

## 硬性约束 (Hard Constraints)

### 1. 静态编译 (NO_DYNAMIC_LOADING)
- **要求**：所有 Backend (CUDA/ROCm/Metal/CPU) 实现必须**静态编译**并链接到主程序。
- **禁止**：使用 `dlopen`, `libloading` 或任何形式的运行时动态库加载。
- **目的**：确保部署原子性，避免 CUDA 运行时版本不匹配导致的 "DLL Hell"。

### 2. 真实性原则 (Ω1: TRUTH)
- **要求**：所有架构/量化/精度信息必须从模型文件自身提供的 **metadata** 读取。
- **禁止**：
  - 基于 **Model ID** 推断架构类型
  - 基于 **文件名** 推测量化类型 (GGUF) 或精度 (ONNX)
  - 使用 `contains()` 模糊匹配进行架构推测
  - 使用硬编码默认值代替模型元数据（如 rope_theta 默认 10000.0，dtype_size 默认 2）
- **目的**：避免错误的推测导致模型加载失败，确保系统行为可预测。
- **关联需求**: REQ-LOADER-014, REQ-LOADER-015, REQ-LOADER-019

**正确做法示例**:
```rust
// ✅ 正确: 从 GGUF 元数据读取
let arch = gguf_metadata.get("general.architecture");

// ✅ 正确: 从 ONNX tensor dtype 读取
let precision = tensor.data_type;

// ✅ 正确: 从 config.json 读取
let arch = config.get("model_type");

// ✅ 正确: 从实际权重张量读取 dtype
let dtype_size = loader.detect_weight_dtype_size()
    .ok_or_else(|| Error::MissingMetadata)?;
```

**错误做法示例**:
```rust
// ❌ 错误: 基于 Model ID 推断
if model_id.contains("llama") { return LlamaArchitecture; }

// ❌ 错误: 基于文件名推测量化类型
if filename.contains("Q4_0") { return Quantization::Q4_0; }

// ❌ 错误: 模糊匹配架构
if token.contains("mistral") { return MistralArchitecture; }

// ❌ 错误: 使用硬编码默认值
let rope_theta = config.get("rope_theta").unwrap_or(10000.0);
let dtype_size = config.get("dtype").unwrap_or(2);  // f16
```

### 3. 量化元数据格式 (ARCH-QUANT-METADATA)

> **关联需求**: REQ-LOADER-014, Ω1 真实性原则
> **实现位置**: `src/loader/mod.rs::QuantizationMetadata`

为了支持完全元数据驱动的量化加载，gllm 定义了 `gllm.quantization` 扩展字段格式，存储在 SafeTensors 文件的元数据中。

#### 元数据格式

```json
{
  "gllm.quantization": {
    "qweight": {
      "bits": 4,
      "signed": false,
      "block_size": 128,
      "companions": {
        "scales": "scales",
        "zeros": "qzeros"
      }
    },
    "qweight_2": {
      "bits": 8,
      "signed": true,
      "block_size": 256,
      "companions": {
        "scales": "scales_2"
      }
    }
  }
}
```

#### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `bits` | `u8` | ✅ | 量化位宽 (4 或 8) |
| `signed` | `bool` | ✅ | 是否为有符号量化 |
| `block_size` | `usize` | ✅ | 量化块大小（不可再使用默认值） |
| `companions.scales` | `String` | ❌ | scales 张量名称模式 |
| `companions.zeros` | `String` | ❌ | zeros 张量名称模式 |

#### 验证规则

1. **bits 必须为 4 或 8**
2. **block_size 必须大于 0**
3. **scales 张量必须存在**（如果指定）
4. **zeros 张量可选**（某些量化格式不需要）

#### Ω1 合规性

- **禁止**基于文件名推断 `bits`（如 `Q4_K_M` → 4 bits）
- **禁止**基于张量形状计算 `block_size`
- **禁止**假设 `signed` 值
- **必须**从 `gllm.quantization` 读取所有量化参数

#### 缺失元数据处理

如果模型文件缺少 `gllm.quantization` 元数据：

```rust
// ❌ 错误: 使用默认值推测
let block_size = metadata.block_size.unwrap_or(128);

// ✅ 正确: 返回明确错误
let metadata = loader.quantization_metadata()?
    .ok_or_else(|| LoaderError::MissingMetadata(
        "模型缺少量化元数据。请在 safetensors 文件中包含 gllm.quantization 字段"
    ))?;
```

#### 已知 Ω1 违规问题（待修复）

**问题1: Phi4 QKV 分割硬编码** (ARCH-QUANT-METADATA-001)
- **位置**: `src/loader/mod.rs::split_phi4_qkv`
- **问题**: 硬编码了 Phi-4-mini 的 Q/K/V 维度 (Q=3072, K=1024, V=1024)
- **当前代码**:
  ```rust
  // Phi-4-mini: Q=3072, K=1024, V=1024
  let q_dim = in_dim;
  let kv_dim = (out_dim - q_dim) / 2;
  ```
- **正确做法**: 应该从 config.json 读取 `num_attention_heads`, `num_key_value_heads`, `head_dim` 来计算
- **状态**: 📋 待修复（需要重构 `maybe_split_fused` 以接收 `ModelConfig`）

---

### 三种推理任务

| 任务 | 描述 | 模型示例 |
|------|------|----------|
| **Embedding** | 文本向量化 | BGE-M3, CodeXEmbed |
| **Rerank** | 文本重排序 | BGE-Reranker |
| **Generator** | 文本生成 | Qwen2, Mistral, MoE |

### 公共 API

**同步/异步客户端**：提供 `Client` 和 `AsyncClient` 两种接口类型

**Builder 模式**：支持链式调用配置生成参数
- `embeddings([texts])` - 批量文本向量化
- `rerank(query, docs)` - 文本重排序
- `generate(prompt).max_tokens(n)` - 文本生成

---

## 架构总览 (Grand Unified Architecture)

为了有效管理多模型、多后端、高性能的复杂需求，系统采用严格的 **4层抽象体系**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 1: Manifest (静态定义层) - "What is it?"                         │
│  组件: ModelManifest (SSOT)                                             │
│  职责: 定义模型身份、下载源、物理架构、默认参数。                       │
│  特性: 纯静态数据 (const), 无逻辑。                                     │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ 驱动
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 2: Adapter (逻辑适配层) - "How to interpret?"                    │
│  组件: ModelAdapter Trait (Qwen3Adapter, Llama4Adapter)                 │
│  职责:                                                                  │
│    1. WeightMapper: 将 safetensors 映射到标准计算图输入。               │
│    2. TokenizerAdapter: 处理 Chat Template (Jinja2/Hardcoded)。         │
│  特性: 无状态 (Stateless), 策略模式。                                   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ 生成
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 3: Engine (运行时调度层) - "When & Where?"                       │
│  组件: Executor, Scheduler (PagedAttention), GraphBuilder               │
│  职责:                                                                  │
│    1. 批处理 (Continuous Batching)。                                    │
│    2. 显存管理 (KV Cache Allocation)。                                  │
│    3. 算子编排 (Operator Graph Recording)。                             │
│  特性: 有状态 (Stateful), 管理资源生命周期。                            │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ 指令 (L3 API)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 4: Driver (硬件执行层) - "Just Do It."                           │
│  组件: gllm-kernels (Backend Trait)                                     │
│  职责:                                                                  │
│    1. CudaBackend (cudarc + AOT CUBIN).                                 │
│    2. CpuBackend (faer + SIMD).                                         │
│  特性: 零拷贝 (Zero-Copy), 物理阻断数据回流, Driver API Only。          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. 模型缓存 (ARCH-MODEL-CACHE)

> **关联需求**: REQ-LOADER-002, REQ-LOADER-005, REQ-LOADER-016

**缓存根目录**: `~/.gllm/models/`

**源头隔离结构** (ARCH-MODEL-CACHE-001):
```
~/.gllm/models/
├── hf/    # HuggingFace 下载的模型
└── ms/    # ModelScope 下载的模型
```

| 源 | 缓存子目录 | 说明 |
|---|-----------|------|
| HuggingFace | `hf/` | HF 专用缓存，使用 hf-hub crate 管理 |
| ModelScope | `ms/` | MS 专用缓存，使用 git-like 结构 |

**隔离原则**:
- 不同下载源的模型文件**互不干扰**
- 即使同名模型，从不同源下载也存储在不同目录
- 禁止跨源混用文件（config 从 HF 读，weights 从 MS 读）

**环境变量**:
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GLLM_CACHE_DIR` | `~/.gllm/models` | 覆盖缓存根目录 |
| `HF_TOKEN` | (空) | HuggingFace 认证 token |

> **自动回退**: HuggingFace 下载失败时会自动切换到 ModelScope，无需手动指定模型来源。

**环境变量铁律 (ARCH-ENV-001)**:
- ❌ AI 禁止擅自引入新的环境变量
- ❌ AI 禁止为"灵活性"或"可配置性"添加环境变量
- ✅ 新增环境变量必须由用户明确要求
- ✅ 环境变量必须有明确的、不可替代的使用场景

---

### 4. 软件工程抽象模式 (ARCH-SOFTWARE-PATTERNS)

Client 端采用以下模式管理模型多样性与调度复杂性：

#### A. 模型适配器模式 (Model Adapter)
将不同模型的加载、分词、后处理差异封装在 Adapter 中，主流程对具体模型无感知。

**接口契约**：
- `supports(manifest)` - 判断适配器是否支持该模型
  - **架构匹配**: `manifest.arch == ModelArchitecture::Qwen3`
  - **用途匹配**: `manifest.kind == ModelKind::Chat`
  - **Ω1 约束**: 禁止基于 Model ID 推断用途（如 `model_id.contains("embed")`）
- `load_weights()` - 差异化加载逻辑（如 Qwen3 Thinking Head）
- `apply_chat_template()` - 差异化分词逻辑（处理 Jinja2/Hardcoded 模板）
- **静态注册表** - 使用常量表存储所有适配器实例

**适配器用途区分 (ModelKind)**：
| ModelKind | 用途 | 适配器示例 |
|-----------|------|-----------|
| `Chat` | 文本生成/对话 | `Qwen3Adapter`, `Llama4Adapter` |
| `Embedding` | 文本向量化 | `Qwen3EmbedAdapter` |
| `Reranker` | 文本重排序 | `Qwen3RerankAdapter` |

#### B. 执行器模式 (Executor Pattern)
将推理调度（Scheduler）与硬件执行（Backend）解耦。

**组件结构**：
- `backend` - 硬件后端实例
- `scheduler` - 调度器实例

**统一入口**：`step()` 方法内部自动处理 PagedAttention 和 Continuous Batching

---

### 5. 运行时模型管理 (ARCH-LIFECYCLE)

为了支持运行时动态切换模型 (Hot Swap)，Client 不再持有静态的 Executor，而是管理其生命周期。

#### 核心机制 (ARCH-LIFECYCLE-SWAP)

**State Container**:
`Client` 内部持有 `Arc<RwLock<Option<Executor>>>`。
- `Arc`: 允许在多线程中克隆 Client。
- `RwLock`: 允许并发读取（推理），但独占写入（切换模型）。
- `Option`: 允许卸载模型进入"空闲状态"。

**切换流程 (Stop-the-World)**:
1. **Acquire Lock**: `client.swap_model()` 请求写锁。这会阻塞所有新的推理请求，并等待当前正在进行的 `generate().next()` 步骤释放读锁。
2. **Drop Old Executor**: 获得写锁后，将 `Option` 置为 `None`。Rust 的 `Drop` 机制自动触发：
   - 释放 `Executor`
   - 释放 `Backend` (CUDA Context / Memory)
   - 释放 `Scheduler` (KV Cache 显存)
   - **关键**: 确保 GPU 显存完全释放。
3. **Load New Model**: 调用 `Loader` 加载新模型权重（此时 GPU 应为空闲）。
4. **Create New Executor**: 初始化新后端，分配新的 KV Cache（基于新模型的维度配置）。
5. **Release Lock**: 将新 Executor 放入 `Option`，释放写锁，恢复服务。

**KV Cache 联动**:
KV Cache 的生命周期严格绑定于 `Executor`。
- **不可复用**: 不同模型的 `head_dim`, `num_layers`, `num_heads` 不同，KV Cache 物理结构不同。
- **全量销毁**: 切换模型必须销毁整个 Cache Pool，重新分配。

---

## 数据流

### GPU 后端数据流（🚨 核心原则）

```
tokens → upload一次 → [GPU全程计算] → readback一次 → 结果
                      ↑                            ↑
                      └──── 全程 GPU 显存 ─────────┘
```

**禁止**：中途 GPU→CPU→GPU 往返

### 模型加载流程

```
Client::new_embedding("BAAI/bge-m3")
    → 检查 ~/.gllm/models/ 是否存在
    → 不存在则 hf-hub 下载
    → SafeTensors 解析权重
    → 初始化模型 → 返回 Client
```

### 3. 模型配置覆盖 (ARCH-MANIFEST)

> **📌 重构中 (REQ-LOADER-006/007/008)**: Manifest 从"模型注册表"重构为"配置覆盖层"

**重构目标**：
- ❌ **旧设计**: 每个模型必须在 `KnownModel` 枚举中注册，硬编码 `hf_repo`、`aliases` 等
- ✅ **新设计**: 用户直接使用 HF Model ID，系统自动下载并识别架构

**重构后流程**：
```
Client::new_chat("Qwen/Qwen3-0.6B")
    ↓
1. 构造 HF 地址: huggingface.co/Qwen/Qwen3-0.6B
2. 下载模型文件 (config.json, tokenizer.json, safetensors)
3. 读取 config.json → 识别架构 (Qwen3)
4. 匹配 Qwen3Adapter → 加载权重
5. (可选) 应用 Manifest 中的配置覆盖
```

**Manifest 重新定位**：

| 旧定位 (模型注册表) | 新定位 (配置覆盖层) |
|-------------------|-------------------|
| 每个模型必须注册 | 仅用于特殊配置覆盖 |
| `model_id` 枚举值 | 移除（由用户输入指定） |
| `aliases` 别名列表 | 移除（用户直接用 HF ID） |
| `hf_repo` 硬编码仓库 | 移除（由用户输入构造） |
| `model_scope_repo` 硬编码 | 移除（自动回退） |
| `arch` 架构类型 | 从 config.json 自动识别 |
| `tensor_rules` 命名规则 | 从架构类型推断 |
| `rope_base_override` | ✅ 保留（特殊配置） |
| `max_context_override` | ✅ 保留（特殊配置） |
| `moe_config` | ✅ 保留（特殊配置） |

**架构识别规则**：

| config.json 字段 | 架构识别 |
|------------------|----------|
| `model_type == "qwen3"` | Qwen3 |
| `model_type == "llama"` | Llama4 |
| `model_type == "ministral"` / `mistral` | Ministral/Mistral3 |
| `model_type == "glm"` | GLM4/GLM5 |
| `architectures[0]["type"]` | 回退字段 |

**约束**：
1. ✅ 支持任意有效的 HF Model ID
2. ✅ 自动回退 ModelScope（HF 失败时）
3. ✅ 新模型无需修改代码即可使用
4. ✅ Manifest 仅用于特殊配置覆盖（如非标准 RoPE base）

---

#### 旧版 Manifest 文档 (重构前)

> **注意**: 以下描述的是旧版 Manifest 系统，正在重构中。

为了实现"只改配置支持新模型"，我们将 HF 标准化(Downloading)和推理配置(Inference)融合为一个统一的 **Model Manifest** 系统，并移除 Registry 依赖。

**核心原则**: `KnownModel` 枚举是整个系统的 SSOT。

**数据结构字段**：

| 类别 | 字段 | 说明 |
|------|------|------|
| **身份与下载** | `model_id` | 内部唯一标识 |
| | `aliases` | 用户别名 (如 "qwen3-7b") |
| | `hf_repo` | HF 仓库 ID (主要源) |
| | `model_scope_repo` | ModelScope 仓库 ID (镜像源) |
| | `hf_file_map` | 关键文件重命名映射 |
| **架构适配** | `arch` | 物理架构 (Llama, Qwen3, Bert...) |
| | `tensor_rules` | 权重命名规则 (前缀/后缀) |
| **推理参数覆盖** | `rope_base_override` | RoPE 基础覆盖值 |
| | `max_context_override` | 最大上下文覆盖值 |
| **MoE 配置** | `num_experts` | 总专家数 |
| | `num_experts_per_tok` | 激活专家数 (Top-K) |
| | `router_type` | 路由算法 |

**约束**：所有逻辑（下载、加载、计算）由静态配置驱动，禁止代码逻辑中散落模型判断。

---

#### 权重加载策略 (ARCH-LOADER)

1.  **架构自适应 (ARCH-LOADER-ADAPTIVE)**
    - Loader 必须根据 `Architecture` 枚举自动选择权重命名策略。
    - 支持前缀探测（如 `bert.` vs 无前缀）。

2.  **文件选择策略 (ARCH-LOADER-FILE-SELECT)**
    - **Ω1 约束**: 禁止基于文件名推测量化类型或精度，禁止"智能排序"
    - **策略**: 首个匹配 (First Match)
      - GGUF: 按字母排序选择第一个 `.gguf` 文件
      - ONNX: 按字母排序选择第一个 `.onnx` 文件
      - SafeTensors: 使用 `model.safetensors.index.json` 中的分片列表
    - **理由**: 模型元数据（量化类型、精度）在文件内部，无需从文件名推测

3.  **融合权重处理 (ARCH-LOADER-FUSED)**
    - 针对 **GPT-OSS** (GPT-2 style) 和 **Qwen3 / Llama 4** 等模型，必须支持 Fused QKV (e.g. `c_attn`, `W_pack`) 和 Fused MLP (e.g. `c_proj`, `gate_up_proj`) 的自动拆分。
    - **禁止**：运行时动态拆分（这会破坏零拷贝原则）。
    - **禁止**：将拆分逻辑下沉至 GPU Backend。Backend 接口应保持数学上的纯净（独立 Q/K/V），格式适配由 Loader 层负责。
    - **要求**：在权重加载阶段（CPU -> GPU Upload 前）完成权重的拆分和重排，确保 GPU 内核接收到的是标准的 Q/K/V 和 Gate/Up/Down 格式。

4.  **并行加载机制 (ARCH-LOADER-PARALLEL)**
    - **瓶颈识别**：串行加载会导致 CPU 计算（权重拆分/重排）和磁盘 I/O 交替阻塞，无法跑满 NVMe 带宽。例如 40GB+ 的 GPT-OSS/MoE 模型，串行加载速度（~700MB/s）远低于 NVMe 理论极限。
    - **策略**：使用并行框架进行层级（Layer-level）并行迭代加载。
    - **约束**：**MoE 模型必须使用并行加载**。因为 MoE 的 Fused Weight 拆分（如 Gate/Up/Down 投影）是 CPU 密集型操作，必须通过并行化来掩盖 I/O 延迟。
    - **核心逻辑**：利用多线程同时触发 Page Fault 和 CPU 处理，每个层独立并行加载。
    - **收益**：多线程同时触发缺页中断（Page Fault），操作系统能更有效地调度磁盘 I/O 队列；同时利用多核 CPU 并行处理 Fused 拆分。

#### ONNX Adapter Architecture (ARCH-ONNX)

**核心架构: Matcher -> Mapper -> Kernel**

ONNX 加载器**严禁**进行 Naive 的 1:1 算子翻译（如将 `MatMul` + `Add` 分别执行）。必须采用图模式匹配（Graph Pattern Matching）将子图融合为高性能算子。

**处理流程**:
1.  **Proto Parsing**: 解析 `.onnx` 文件，构建原始计算图。
2.  **Pattern Matching**: 使用子图同构算法识别融合模式（如 `MatMul` + `Add` + `Gelu`）。
3.  **Kernel Mapping**: 将匹配到的子图映射为单一的 `FusedKernel` 调用（如 `GemmGelu`）。
4.  **Fallback**: 仅当无法匹配任何模式时，降级为 Atomic Kernel 执行。

**关键融合模式**:
- `Linear` + `Bias` -> `FusedGemm`
- `Gemm` + `Gelu`/`Silu` -> `FusedGemmAct`
- `Attention` Subgraph -> `FlashAttention`
- `LayerNorm` Subgraph -> `FusedLayerNorm`

### 推理流程

```
client.embeddings(["text"]).generate()
    → Tokenizer 编码
    → Backend::embedding()      # GPU
    → Backend::attention_block() # GPU (N 层)
    → Backend::ffn_block()       # GPU (N 层)
    → Backend::mean_pooling()    # GPU
    → Backend::normalize()       # GPU
    → readback → 返回结果
```

---

## 目录结构

```
src/
├── lib.rs           # 公共 API 导出
├── client.rs        # Client / AsyncClient
├── embeddings.rs    # Embeddings API
├── rerank.rs        # Rerank API
├── engine.rs        # 推理引擎
├── generation.rs    # 生成循环
├── kv_cache.rs      # KV Cache
├── weight_loader.rs # 权重加载
└── model_config.rs  # 模型配置
```

---

## 存储结构

```
~/.gllm/models/
├── BAAI--bge-m3/
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
└── Qwen--Qwen3-7B/
    └── ...
```

---

## 2026 Accuracy First 架构 (ARCH-ACCURACY)

> **核心哲学**: 在 2026 年标准下，推理引擎的首要任务是保证**数值确定性 (Determinism)** 和 **比特级可复现性 (Bitwise Reproducibility)**。任何牺牲精度的吞吐量优化（如 vLLM 的乱序执行）在 gllm 中默认禁用。

### 1. 确定性调度 (ARCH-ACCURACY-SCHED)

**问题**: 浮点数加法不满足结合律 (`(a+b)+c != a+(b+c)`)。在 GPU 并行计算中，Batch 内请求的物理布局顺序变化会导致 Attention 规约顺序变化，从而产生微小的数值漂移。

**解决方案**: **Canonical Ordering (规范序)**
- **约束**: 调度器输出的 `ScheduledBatch` 必须严格按照 `RequestId` (或创建时间戳) 排序。
- **禁止**: 严禁为了填充空隙（Bin Packing）而打乱请求在 Batch 中的物理顺序。
- **收益**: 无论系统负载如何波动，同一组请求在 Batch 中的相对位置永远固定，消除内存布局引起的误差。

### 2. 串行微批次执行 (ARCH-ACCURACY-EXEC)

**问题**: 即使 Layout 固定，大 Batch 的 GPU Kernel 内部并行规约路径仍可能因硬件调度产生不确定性（Non-Batch-Invariant）。

**解决方案**: **Micro-Batch Serial Execution**
- **机制**: 在 `Executor::step()` 内部，不将整个 Batch 打包为一个大 Tensor 发送给 GPU。
- **实现**: 采用 Rust 循环，**串行**处理 Batch 中的每个请求（或极小的 Micro-Batch）。
- **权衡**: 牺牲 Kernel Launch 开销（吞吐量下降），换取数学上的绝对正确性。
- **配置**: **强制启用**。移除任何切换回并行模式的配置项。

### 3. 阶段隔离 (ARCH-ACCURACY-ISOLATION)

**问题**: Prefill（计算密集）和 Decode（带宽密集）混合会导致计算图剧烈抖动，影响算子精度。

**解决方案**: **Strict Phase Isolation**
- **规则**: 一个 Batch 必须是 **纯 Prefill** 或者 **纯 Decode**。
- **废弃**: `Chunked Prefill` (ARCH-SCHED-CHUNKED) 已被永久废弃。
- **优先级**: 调度器优先处理 Decode 队列。仅当 Decode 队列为空时，才处理 Prefill 队列。

---

## 调度器架构 (ARCH-SCHED)

> **详细设计**: 见 [SPEC/DOCS/scheduling/hgal-scheduler-algorithm.md](./DOCS/scheduling/hgal-scheduler-algorithm.md)
> **📌 SSOT**: 后端架构约束见 [gllm-kernels/SPEC/02-ARCHITECTURE.md](../gllm-kernels/SPEC/02-ARCHITECTURE.md)

### 调度器定位 (ARCH-SCHED-001)

调度器位于 Layer 3 (Engine)，负责：
1. **PagedAttention 管理**: 页面分配、换出、换入
2. **Continuous Batching**: 动态批处理，序列加入/移除
3. **KV Cache 管理**: GPU/CPU 内存协调

### 核心约束 (ARCH-SCHED-CONSTRAINTS)

| 约束 | 说明 | 违规后果 | SSOT位置 |
|------|------|----------|----------|
| **禁止序列内页面分散** | 必须以序列组 (SequenceGroup) 为单位换出 | 序列错乱、内存碎片 | - |
| **禁止新换入页立即换出** | Warm-up 保护期内禁止换出 | Cache Thrashing | - |
| **禁止纯 LRU** | 必须使用 LIRS 或 CLOCK-Pro 算法 | 无法区分访问模式 | - |
| **零拷贝原则** | Swap 操作不介入生成循环数据流 | 违反 L3 GPU-Pure | gllm-kernels §L3 GPU-Pure |
| **AOT CUBIN** | 禁止动态 kernel 编译 | 违反 AOT 策略 | gllm-kernels §AOT Only |
| **Driver API Only** | 仅依赖 libcuda.so | 运行时依赖问题 | gllm-kernels §Driver API Only |

### 数据结构约束 (ARCH-SCHED-DATA)

| 结构 | 约束 | 说明 |
|------|------|------|
| **PageState** | 必须包含 Warm/Protected 状态 | 支持 Warm-up 和 Working Set |
| **SequenceGroup** | 必须包含 pages 列表 | Gang-Aware 换出 |
| **PageMetadata** | 必须包含 recency/access_count | LIRS 优先级计算 |

### API 约束 (ARCH-SCHED-API)

调度器必须提供以下接口：

| 接口 | 功能 |
|------|------|
| `select_victim_groups(count: usize) -> Vec<RequestId>` | Gang-Aware 受害者选择 |
| `handle_page_fault(page_id, sequence_id) -> PageLocation` | 页面错误处理 |
| `get_batch_page_table(request_ids) -> BatchPageTable` | 批次页表获取 |
| `update_batch(batch, results) -> BatchAction` | 批次状态更新 |

### 禁止行为 (ARCH-SCHED-PROHIBITED)

| 禁止项 | 原因 | 检测方法 |
|--------|------|----------|
| 禁止单页面换出 | 破坏 Gang Scheduling | 检查返回值类型 |
| 禁止在生成循环中 Swap | 违反零拷贝原则 | 检查 Swap 调用位置 |
| 禁止硬编码换出阈值 | 必须动态计算 | 检查魔法数字 |
| **禁止 while let 无限循环** | **CPU 100% 死锁** | **检查 admit_waiting 实现** |

### 企业级死锁防护 (ARCH-SCHED-DEADLOCK-FREE)

> **关联需求**: REQ-CORE-004 (精度优先架构)

**问题**: 如果 `admit_waiting` 使用 `while let` 循环处理等待队列，当所有序列都无法分配内存时会导致无限循环，CPU 100%。

**解决方案** (drain 模式):
```rust
fn admit_waiting(&mut self, scheduler: &mut PagedScheduler) {
    // 1. 使用 drain(..) 一次性收集所有等待序列
    let waiting_sequences: Vec<_> = self.waiting.drain(..).collect();

    // 2. 尝试 admit 每个序列
    for mut sequence in waiting_sequences {
        match scheduler.add_sequence(sequence.to_sequence_group()) {
            Ok(()) => {
                // 成功：加入 running 队列
                self.running.insert(sequence.id, sequence);
            }
            Err(_) => {
                // 失败：放回 waiting 队列末尾，等待下次 build_batch 重试
                // 关键：不在本次调用中重试，避免无限循环
                self.waiting.push_back(sequence);
            }
        }
    }
}
```

**关键设计点**:
1. **`drain(..)`** - 一次性清空等待队列，避免在迭代过程中修改队列
2. **失败序列放回末尾** - 下次 `build_batch` 时会重新尝试
3. **本次调用不重试** - 避免无限循环，确保系统始终能响应新请求

**验收标准**:
- CPU 不会因内存不足而 100% 占用
- 系统在 OOM 时仍能接受新请求（即使无法立即处理）
- 失败的序列会在后续批次中自动重试

---

## 2024 vLLM 优化 (ARCH-SCHED-2024)

> **兼容性声明**: 所有以下优化均为**调度/算法级别**改进，与 **AOT CUBIN 策略**完全兼容。无需修改内核编译流程。

### 概述 (ARCH-SCHED-2024-OVERVIEW)

基于 vLLM 2024 Q2 Roadmap 和最新论文，引入三项 PagedAttention 性能优化：

| 优化 | 核心收益 | 论文/来源 | AOT 兼容性 |
|------|----------|-----------|------------|
| **Chunked Prefill / SplitFuse** | 消除 Prefill-Decode 阶段隔离，提升吞吐 30-50% | vLLM #3861 | ✅ 纯调度优化 |
| **SwiftKV** | Prefill 计算减少 50%，KV Cache 减少 62.5% | arXiv:2410.03960 | ✅ 算法级优化 |
| **LMCache** | 跨请求 KV Cache 共享，重复提示吞吐提升 15× | LMCache GitHub | ✅ 缓存层优化 |

---

### 1. Chunked Prefill / SplitFuse (ARCH-SCHED-CHUNKED) [DEPRECATED]

> **🔴 状态**: 已废弃。与 2026 Accuracy First 架构中的 Phase Isolation 冲突。

#### 1.1 问题定义 (旧)

传统 Continuous Batching 将 Prefill（填充）和 Decode（解码）阶段完全隔离：

```
传统方式（阶段隔离）：
Batch 1: [Prefill-512, Prefill-256] → 完成后清空
Batch 2: [Decode-1, Decode-1, Decode-1, ...] → 纯 Decode
```

**问题**：
1. Prefill 阶段 Decode 请求必须等待，导致 Tail Latency 恶化
2. GPU 利用率在纯 Prefill 阶段（Memory Bound）和纯 Decode 阶段（Compute Bound）之间波动
3. 批次切换开销大

#### 1.2 Chunked Prefill 设计

**核心思想**：将长 Prefill 请求切分为多个 Chunk，与 Decode Token 交织调度。

```
Chunked 方式（交织调度）：
Batch 1: [Prefill-64, Decode-1, Decode-1, Decode-1, Prefill-64, ...]
         ↑ Chunk 1      ↑ Decode 插槽    ↑ Chunk 2
```

**算法流程**：

```
1. 请求分类
   ├─ Prefill 请求: 按 chunk_size 切分 (如 64/128 tokens)
   └─ Decode 请求: 每次 1 token

2. 批次构建 (每次调度)
   ├─ 优先: Decode 请求 (保证低延迟)
   ├─ 剩余插槽: 填入 Prefill Chunk
   └─ 约束: total_tokens ≤ max_batch_tokens

3. 状态跟踪
   └─ PrefillRequest.pending_chunks > 0 → 继续调度
```

#### 1.3 SplitFuse 优化

进一步细分 Prefill 阶段：

```
SplitFuse = 分离 Q/K/V 计算 + 融合 Attention

阶段 A (分离计算，可并行):
  ├─ Prefill-Chunks[N].Q_proj
  ├─ Prefill-Chunks[N].K_proj
  └─ Prefill-Chunks[N].V_proj

阶段 B (融合 Attention，顺序执行):
  └─ FlashAttention(所有已完成的 Chunk)
```

**收益**：阶段 A 可充分利用 GPU Tensor Core 并行，阶段 B 利用 FlashAttention 的内存优化。

#### 1.4 架构约束

| 约束 | 说明 | 违规后果 |
|------|------|----------|
| **Chunk Size 必须对齐** | 必须是 page_size 的整数倍 | 导致页面边界错误 |
| **禁止 Chunk 间数据依赖** | 每个 Chunk 必须独立可计算 | 无法并行 |
| **KV Cache 原子写入** | 单个 Chunk 的 KV Cache 必须完整写入 | 数据竞争 |
| **AOT CUBIN 兼容** | 不得使用动态 kernel 编译 | 违反 ARCH-AOT-CUBIN |

#### 1.5 API 接口

**配置结构**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunk_size` | 数值 | Chunk 大小 (默认 64 tokens) |
| `decode_slots` | 数值 | 每个 Batch 保留的 Decode 插槽数 |
| `enable_splitfuse` | 布尔值 | 是否启用 SplitFuse |

**请求类型**：
- `Prefill` - 包含 total_tokens, completed_chunks, pending_chunks
- `Decode` - 单 token 解码

**调度器方法**：
- `build_chunked_batch()` - 构建交织批次
  1. 收集所有待处理的 Decode 请求
  2. 计算剩余 token 预算
  3. 填入 Prefill Chunk
- `update_chunk_progress()` - 更新 Chunk 状态

---

### 2. SwiftKV 算法 (ARCH-SCHED-SWIFTKV)

> **论文**: SwiftKV: Efficient KV Cache Compression for LLM (arXiv:2410.03960)

#### 2.1 问题定义

传统 PagedAttention 将所有 KV Cache 全量保存：

```
传统方式 (Full KV):
Layer 0: [K0, V0, K1, V1, K2, V2, ..., K511, V511]  # 1024 向量
Layer 1: [K0, V0, K1, V1, K2, V2, ..., K511, V511]  # 1024 向量
...
```

**问题**：
1. 内存占用随序列长度线性增长
2. 后续 Token 的 KV 贡献度递减（Attention 权重衰减）
3. 大量 KV Cache 用于低 Attention Score 的 Token

#### 2.2 SwiftKV 核心技术

**SingleInputKV (SIKV)**: 单输入 KV 蒸馏

```
核心思想: 将连续的 N 个 KV 向量蒸馏为 1 个

算法:
  1. 将序列按窗口大小 W 分组 (如 W=4)
  2. 每组内: [K0, K1, K2, K3] → Attention 加权 → [K_merged]
  3. 只保留 merged KV

效果:
  - KV Cache 数量: N → N/W (减少 75% 当 W=4)
  - 计算复杂度: O(N²) → O(N²/W)
```

**AcrossKV (AKV)**: 跨层 KV 共享

```
核心思想: 相邻层的 KV Cache 高度相关，可以共享

算法:
  1. 计算 Layer[i] 和 Layer[i-1] 的 KV 相似度
  2. 高相似度 (>0.9) → 共享前一层的 KV
  3. 低相似度 → 保留当前层 KV

效果:
  - KV Cache 数量: 进一步减少 50%
  - 精度损失: <0.1% (Perplexity)
```

#### 2.3 架构约束

| 约束 | 说明 | 违规后果 |
|------|------|----------|
| **蒸馏只在 CPU 端** | KV 蒸馏在 Swap-out 时执行 | 避免额外 GPU 计算 |
| **可配置蒸馏率** | 用户可选择 W=2/4/8 | 兼顾精度与性能 |
| **AOT CUBIN 兼容** | Attention 内核无需修改 | 算法层透明优化 |
| **精度回退** | 检测到精度损失时自动禁用 | 保证正确性 |

#### 2.4 API 接口

**SwiftKV 配置**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled` | 布尔值 | 是否启用 SwiftKV |
| `window_size` | 数值 | SIKV 窗口大小 (2/4/8) |
| `enable_across_kv` | 布尔值 | 是否启用 AKV |
| `similarity_threshold` | 浮点数 | AKV 相似度阈值 |
| `precision_guard` | 浮点数 | 精度损失阈值 (PPL 差值) |

**SwapManager 扩展方法**：
- `swap_out_with_distillation()` - Swap-out 时执行蒸馏
  1. 检查是否启用 SwiftKV
  2. 执行 SIKV 蒸馏（合并连续页面）
  3. 执行 AKV 跨层共享（可选）
  4. Swap-out 压缩后的页面
- `distill_kv_pages()` - KV 蒸馏 (SIKV)
- `share_across_layers()` - 跨层共享 (AKV)

---

### 3. LMCache 跨请求 KV Cache (ARCH-SCHED-LMCACHE)

> **项目**: LMCache - A Distributed KV Cache for LLM Serving (GitHub: LMCache/LMCache)

#### 3.1 问题定义

多用户场景下，大量请求包含相同的系统提示或上下文：

```
请求 1: [SystemPrompt (512 tokens) + UserQuery (10 tokens)]
请求 2: [SystemPrompt (512 tokens) + UserQuery (15 tokens)]
请求 3: [SystemPrompt (512 tokens) + UserQuery (8 tokens)]
```

**问题**：相同的 SystemPrompt KV Cache 被重复计算 3 次。

#### 3.2 LMCache 核心设计

**三层缓存架构**：

```
┌─────────────────────────────────────────────────────────────────┐
│  L1: GPU In-Memory Cache (最快，最小)                          │
│  ─────────────────────────────────────────                     │
│  存储: 当前活跃请求的 KV Cache                                  │
│  命中率: ~5% (仅当前 Batch 重复)                                │
│  容量: ~100MB (显存限制)                                        │
├─────────────────────────────────────────────────────────────────┤
│  L2: CPU Offload Cache (中等，中量)                            │
│  ─────────────────────────────────────────                     │
│  存储: 最近使用的 KV Cache (LRU 淘汰)                           │
│  命中率: ~30% (跨 Batch 重复)                                   │
│  容量: ~10GB (系统内存)                                         │
├─────────────────────────────────────────────────────────────────┤
│  L3: Distributed Cache (慢，无限)                               │
│  ─────────────────────────────────────────                     │
│  存储: Redis/S3/本地磁盘                                        │
│  命中率: ~65% (跨会话/跨实例重复)                               │
│  容量: 无限                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Cache Key 设计**：

```
CacheKey = Hash(
    model_id +
    prompt_prefix_tokens +  // 只缓存前 N tokens (如前 512)
    layer_indices           // 哪些层的 KV Cache
)

示例:
  SystemPrompt = "You are a helpful assistant..."
  → Tokenized: [101, 102, ..., 612] (512 tokens)
  → Hash: SHA256(model_id + tokens) → "abc123..."
```

#### 3.3 跨请求命中流程

```
新请求: [SystemPrompt(512) + UserQuery(10)]

1. 计算 CacheKey
   key = hash(model_id + SystemPrompt_tokens)

2. L1 查询 (GPU)
   └─ Miss → 继续 L2

3. L2 查询 (CPU)
   ├─ Hit → DMA 复制到 GPU → 跳过前 512 tokens 的 Prefill
   └─ Miss → 继续 L3

4. L3 查询 (分布式)
   ├─ Hit → 加载到 L2 → DMA 复制到 GPU
   └─ Miss → 正常 Prefill，结果写入 L1/L2/L3

收益:
  └─ 512 tokens Prefill → 0 (命中时)
  └─ 只需 Prefill 新增的 10 tokens
```

#### 3.4 架构约束

| 约束 | 说明 | 违规后果 |
|------|------|----------|
| **Cache Key 唯一性** | 相同输入必须产生相同 Key | 缓存污染 |
| **版本一致性** | 模型权重更新时 Cache 失效 | 返回错误结果 |
| **AOT CUBIN 兼容** | DMA 复制使用现有 Memcpy 内核 | 违反 ARCH-AOT-CUBIN |
| **禁止生成循环中 L3 访问** | L3 访问必须在批次间执行 | 违反零拷贝原则 |
| **原子写入** | Cache Entry 必须完整写入后可见 | 部分数据导致错误 |

#### 3.5 API 接口

**LMCache 配置**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `l1_capacity_mb` | 数值 | GPU L1 容量 |
| `l2_capacity_mb` | 数值 | CPU L2 容量 |
| `l3_backend` | 枚举 | Redis/LocalDisk/Disabled |
| `cache_prefix_len` | 数值 | 只缓存前 N tokens |

**LMCache 管理方法**：
- `get(key, backend)` - 查询缓存（三层查找：L1→L2→L3）
  - L1 命中：直接返回 GPU KV
  - L2 命中：DMA 复制到 GPU，回填 L1
  - L3 命中：加载到 L2，DMA 复制到 GPU，回填 L1/L2
- `put(key, kv_cache, backend)` - 写入缓存（Prefill 完成后）
  - 写入 L1
  - 异步写入 L2 和 L3
- `invalidate_model(model_id)` - 失效缓存（模型更新时）

**调度器集成**：
- `schedule_with_cache(request, lmcache)` - 缓存感知调度
  1. 计算缓存键
  2. 查询缓存
  3. 命中时跳过 Prefill，未命中时正常执行并写入缓存

#### 3.6 全局内存管理器 (ARCH-SCHED-GLOBAL-MEM)

**核心抽象**：虚拟内存 → 物理内存映射

| 数据结构 | 说明 |
|---------|------|
| `VirtualPageId` | 虚拟页标识 `(sequence_id, logical_index)` |
| `PageLocation` | 物理页位置 `(physical_id, tier)` |
| `PageTable` | 虚拟 → 物理映射表 `HashMap<VirtualPageId, PageLocation>` |
| `Tier` | 存储层级 `L1(GPU) / L2(CPU) / L3(Disk)` |

**核心操作**：
- `allocate_virtual(sequence, num_pages)` → 返回虚拟页范围
- `map_to_physical(virtual_page, tier)` → 建立映射
- `evict(tier, policy)` → 按策略驱逐页
- `swap_tier(virtual_page, from_tier, to_tier)` → 跨层迁移

**驱逐策略 (EvictionPolicy)**：

| 因子 | 计算方式 | 权重 |
|------|----------|------|
| Recency | `current_time - last_access` | 高 |
| Frequency | `access_count / age` | 中 |
| Semantic | 基于 HGAL 的注意力分数 | 低 (Phase 2) |

**多层级管理**：

| Tier | 介质 | 容量 | 访问延迟 | 用途 |
|------|------|------|----------|------|
| L1 | GPU HBM | 有限 | 极低 | 活跃计算 |
| L2 | CPU RAM (Pinned) | 大 | 中等 (PCIe) | 交换缓冲 |
| L3 | NVMe SSD | 巨大 | 高 | 冷数据 |

**企业级约束**：
- **OOM 保护**: L1 满时主动 Swap-out，而非崩溃
- **原子操作**: 跨 Tier 迁移必须是原子的
- **一致性**: 页表更新必须与物理迁移同步
- **可恢复性**: L3 数据支持进程重启后恢复

---

### 4. 三项优化集成架构 (ARCH-SCHED-INTEGRATION)

#### 4.1 数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│  新请求到达                                                         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  LMCache 查询       │
                    │  (L1 → L2 → L3)     │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              │ 命中                         │ 未命中
              ▼                               ▼
        ┌───────────┐                 ┌──────────────┐
        │ Cache Hit │                 │ Chunked      │
        │ (跳过     │                 │ Prefill      │
        │  Prefix)  │                 │ (交织调度)   │
        └─────┬─────┘                 └──────┬───────┘
              │                               │
              ▼                               ▼
        ┌─────────────┐              ┌────────────────┐
        │ Decode      │              │ SwiftKV 蒸馏   │
        │ Loop        │              │ (Swap-out 时)  │
        └─────────────┘              └────────┬───────┘
                                             │
                                             ▼
                                    ┌────────────────┐
                                    │ KV Cache 写入  │
                                    │ (L1/L2/L3)     │
                                    └────────────────┘
```

#### 4.2 配置统一

**Scheduler2024 配置结构**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunked` | ChunkedConfig | Chunked Prefill / SplitFuse 配置 |
| `swift_kv` | SwiftKVConfig | SwiftKV 算法配置 |
| `lmcache` | LMCacheConfig | LMCache 缓存配置 |
| `enable_2024_optimizations` | 布尔值 | 全局开关 |

#### 4.3 验收标准

| 优化 | 验收指标 | 测试场景 |
|------|----------|----------|
| **Chunked Prefill** | Tail Latency (P99) < 50ms vs 纯 Decode Batch | 混合 Prefill/Decode 负载 |
| **SwiftKV** | KV Cache 减少 50%+, 精度损失 < 0.1% PPL | 长序列推理 (32k+) |
| **LMCache** | 重复提示吞吐提升 10×+, Cache 命中率 > 70% | 相同 SystemPrompt 多请求 |

---

### 5. AOT CUBIN 兼容性声明 (ARCH-SCHED-AOT-COMPAT)

> **重要**: 以上三项优化均为 **调度/算法/缓存级别** 改进，不涉及内核编译变更。

| 组件 | 是否需要新 Kernel | 理由 |
|------|-------------------|------|
| Chunked Prefill | ❌ | 仅改变调度逻辑，内核调用不变 |
| SplitFuse | ❌ | Q/K/V 分离计算使用现有 Linear 内核 |
| SwiftKV | ❌ | 蒸馏在 CPU 端或 Swap 时执行 |
| LMCache | ❌ | DMA 复制使用现有 Memcpy 内核 |

**结论**: ✅ **完全兼容 AOT CUBIN 策略，无需修改内核编译流程。**
