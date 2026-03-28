# gllm 架构设计

## 定位

**gllm = 推理客户端库**

负责模型管理、配置、分词、权重加载，调用 gllm-kernels 后端执行计算。

---

## 硬性约束 (Hard Constraints)

### 1. 静态编译 (NO_DYNAMIC_LOADING)
- **要求**：gllm 自身的所有模块必须**静态编译**并链接到主程序。
- **禁止**：gllm 层使用 `dlopen`, `libloading` 或任何形式的运行时动态库加载。
- **例外**：gllm-kernels GPU 后端通过 `dlopen` 加载 GPU driver API（`libcuda.so` / `libamdhip64.so` / `Metal.framework`），这是 gllm-kernels 的内部实现细节，不违反本约束。gllm 不直接调用 `dlopen`。
- **目的**：确保 gllm 部署原子性。GPU driver 由操作系统/驱动安装提供，不属于应用依赖。

### 2. 真实性原则 (Ω1: TRUTH)
- **要求**：所有架构/量化/精度信息必须从模型文件自身提供的 **metadata** 读取。
- **禁止**：
  - 基于 **Model ID** 推断架构类型
  - 基于 **文件名** 推测量化类型 (GGUF) 或精度 (ONNX)
  - 使用 `contains()` 模糊匹配进行架构推测
  - 使用硬编码默认值代替模型元数据（如 rope_theta 默认 10000.0，head_dim 默认 128）
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

// ✅ 正确: 从实际权重张量形状推导 head_dim
let head_dim = loader.detect_head_dim_from_tensor()
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
let head_dim = config.get("head_dim").unwrap_or(128);
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

#### 已知 Ω1 违规问题

**问题1: Phi4 QKV 分割硬编码** (ARCH-QUANT-METADATA-001)
- **状态**: 🟢 已修复 — `split_phi4_qkv` 函数已在架构重构中移除，Phi4 现在走 YAML 模板驱动，维度从 `ModelConfig` 动态读取

---

### 三种推理任务

| 任务 | 描述 | 模型示例 |
|------|------|----------|
| **Embedding** | 文本向量化 | BGE-M3/M4, Qwen3-Embed, E5, Jina v4 |
| **Rerank** | 文本重排序 | BGE-Reranker-v3, Qwen3-Rerank |
| **Generator** | 文本生成 | Qwen3, Llama 4, GLM-5, Mistral 3, Phi-4 |

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
│    2. TokenizerAdapter: 提供 Tokenizer 实例和编码/解码接口。           │
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
│    1. CpuBackend (JIT 编译器: 标量→符号执行→融合→SIMD 代码生成).        │
│    2. GpuBackend (JIT 编译器: 同一管线, Phase 3 生成 PTX/AMDGPU/AIR).   │
│  特性: 零拷贝 (Zero-Copy), 物理阻断数据回流, Driver API Only。          │
│  桥接: `src/compat.rs` 提供类型兼容层，统一 gllm-kernels 导出路径。     │
│  量化: Backend trait 提供 quantized_matmul/dequantize 默认方法，          │
│        CpuBackend 实现分发到 gllm-kernels K-Quant/Classic/IQ 三族。      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. 模型缓存 (ARCH-MODEL-CACHE)

> **关联需求**: REQ-LOADER-002, REQ-LOADER-005, REQ-LOADER-016

**缓存根目录**: `~/.gllm/models/`

**源头隔离结构** (ARCH-MODEL-CACHE-001):
```
~/.gllm/models/
├── huggingface/    # HuggingFace 下载的模型
└── modelscope/     # ModelScope 下载的模型
```

| 源 | 缓存子目录 | 说明 |
|---|-----------|------|
| HuggingFace | `huggingface/` | HF 专用缓存，使用 hf-hub crate 管理 |
| ModelScope | `modelscope/` | MS 专用缓存，使用 git-like 结构 |

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

5.  **通用张量抽象 (ARCH-LOADER-003)**
    - **目的**: 摆脱对特定 Model Adapter 的依赖，实现 "Universal Loader"。
    - **TensorProvider**: 统一 SafeTensors/ONNX/GGUF 的底层接口，屏蔽文件格式差异。
    - **TensorRole**: 引入语义化角色（Embedding, AttentionQuery, FfnGate 等），通过 Regex 规则表将物理张量名映射到逻辑角色。
    - **Shape Heuristics**:
        - `vocab_size` = `Embedding` 张量的较大维度
        - `head_dim` = `AttentionQuery` 张量维度 / `num_heads`
        - `num_layers` = 匹配到的最大 `layer_idx` + 1
    - **DType Adapter**: **(已弃用)** Loader 层不再根据硬件进行动态类型分发。所有输入类型的张量统一在 Load-time 转换为目标的静态 TurboQuant 位宽，抹平原始格式差异。

6.  **量化与降维处理 (ARCH-LOADER-TURBO-QUANT)**
    - **目的**: 彻底消灭推理期的多态执行。所有张量在装载时经过极化转换，统一成 JIT 内核接受的固定底层位宽（如 INT4）。
    - **Load-time Annihilation**: 装载时不再保留原始浮点类型，强行根据 `turbo_quant_bits` 执行 `PolarQuant` 映射。
    - **Native Float 摒除**: 不再有 Native Float 分流。即使原文件是 F16/BF16，统一并入静态编译位宽处理管线，将精度差异化在加载期全部抹平。
    - **Backend::dequantize 废除**: 不再提供反量化能力。JIT 内核生成的汇编直接读取压扁的量子网格点，数学运算纯基于整数/微字节累加器进行，实现零分支执行。

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
├── lib.rs              # 公共 API 导出
├── compat.rs           # 兼容层: 桥接 gllm-kernels 类型 (Backend, CpuBackend, Element 等)
├── client.rs           # Client / AsyncClient
├── embeddings.rs       # Embeddings API
├── rerank.rs           # Rerank API
├── generation.rs       # 生成循环
├── tokenizer.rs        # Tokenizer 集成
├── model_config.rs     # 张量驱动模型配置 (Ω1)
├── weight_loader.rs    # 权重加载工具
├── kv_cache.rs         # KV Cache 结构
├── quantization.rs     # 量化支持
├── loader/             # 模型加载 (HF/MS/GGUF/ONNX/SafeTensors)
│   ├── mod.rs          # 统一加载入口
│   ├── gguf/           # GGUF 解析器 (零拷贝)
│   ├── onnx/           # ONNX protobuf 解析器
│   └── ...             # downloader, format_detector, adapter 等
├── arch/               # 架构 YAML 模板 → OnnxGraph
├── graph/              # DAG 优化器 (统一表示)
│   └── optimizer/      # 优化 Pass (模式融合/硬件融合/DCE)
├── engine/             # 推理引擎 (executor, pipeline)
├── scheduler/          # 调度器 (HGAL, PagedAttention, Continuous Batching)
├── backend/            # 后端检测与降级 (CUDA→CPU)
└── manifest/           # 模型 Manifest 类型
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
- **补充**: 允许在 Prefill 阶段使用 `ChunkedConfig` 时间片切分，但不允许与 Decode 混批。
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
| **零拷贝原则** | Swap 操作不介入生成循环数据流 | 数据回流到 CPU | gllm-kernels SPEC/02 ARCH-SCOPE |
| **JIT 统一路径** | GPU kernel 由 JIT 编译器生成（PTX/AMDGPU/AIR） | 违反 ARCH-JIT-FIRST | gllm-kernels SPEC/04 §1 |
| **Driver API Only** | 仅依赖 libcuda.so / libamdhip64.so / Metal.framework | 运行时依赖问题 | gllm-kernels SPEC/04 §5 |

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

## 调度器重构架构 (ARCH-SCHED-REFACTOR-2026)

> **设计来源**: `.serena/memories/scheduler_refactor_design.md`
> **目标**: 删除 `vllm2024.rs` 中冗余 LMCache，实现准确度优先、确定性优先的 KV 复用架构。

### 重构总览 (ARCH-SCHED-REFACTOR-OVERVIEW)

| 组件 | 目标 | 关键约束 |
|------|------|----------|
| `KvPrefixIndex` | 无 Session 场景下的跨请求前缀复用 | O(n) 最长前缀匹配，禁止 hash-only 全量相等匹配 |
| `SessionKvCache` | AI 编程场景的会话级确定性复用 | 仅允许 append 前缀 claim，禁止越界复用 |
| `KvPipeline` | Thinking 模型双管线隔离 | `Working` 可丢弃，`Conversation` 可跨轮保留 |
| `BatchOrderPolicy` | 批处理确定性顺序 | 默认 `StrictRequestIdOrder`，准确度 > 吞吐量 |

### 1. KvPrefixIndex 前缀树索引 (ARCH-SCHED-PREFIX-INDEX)

**问题**: Hash 缓存只能命中完全相同 token 序列，无法识别 append 关系。

**方案**: 引入 token 前缀树索引，查找最长可复用前缀。

```rust
pub struct KvPrefixIndex {
    root: TrieNode,
}

struct TrieNode {
    children: HashMap<TokenId, TrieNode>,
    page_ref: Option<(VirtualPageId, usize)>,
}
```

**集成约束**:
1. `GlobalMemoryManager` 负责维护索引生命周期，与页表一致更新。
2. `prepare_prefill_with_auto_reuse(request_id, tokens)` 必须先做最长前缀匹配，再决定是否分配新页面。
3. 前缀命中后采用虚拟页映射复用语义，禁止直接共享可写物理页（避免污染）。

### 2. SessionKvCache 会话级复用 (ARCH-SCHED-SESSION-KV)

**问题**: AI 编程会话是确定性 append 流，跨请求前缀可预测。

**方案**: 引入显式 Session KV 结构，保存会话已确认前缀边界。

```rust
pub struct SessionKvCache {
    session_id: SessionId,
    pages: Vec<VirtualPageId>,
    finalized_position: usize,
}
```

**集成约束**:
1. `register_session(session_id)` 创建会话状态。
2. `claim_session_prefix(session_id, request_id, prefix_tokens)` 只能 claim `finalized_position` 范围内页面。
3. `finalize_session_tokens(session_id, new_finalized_position)` 只能单调递增。

### 3. KvPipeline 多管线分离 (ARCH-SCHED-PIPELINE)

**问题**: Thinking/Reasoning 过程不应污染跨轮会话缓存。

**方案**: 将虚拟页命名空间扩展为双管线。

```rust
pub enum KvPipeline {
    Conversation,
    Working,
}

pub struct PipelinedVirtualPageId {
    pub pipeline: KvPipeline,
    pub sequence_id: RequestId,
    pub logical_index: usize,
}
```

**集成约束**:
1. `Working` 管线仅用于本轮中间推理，可在 `prepare_next_turn` 全量释放。
2. `Conversation` 管线用于跨轮上下文保留，不受 `Working` 释放影响。
3. 页表、换出策略、预取逻辑均必须按 `pipeline` 维度隔离。

### 4. BatchOrderPolicy 确定性批处理 (ARCH-SCHED-BATCH-ORDER)

**问题**: 吞吐优先重排会改变浮点规约顺序，破坏可复现性。

**方案**: 批构建阶段引入顺序策略并默认严格 RequestId 排序。

```rust
pub enum BatchOrderPolicy {
    StrictRequestIdOrder,
    FifoOrder,
}
```

**集成约束**:
1. 默认策略必须为 `StrictRequestIdOrder`。
2. `Executor::run_batch_forward` 需校验输入序列严格单调递增。

### 5. GlobalMemoryManager 融合策略 (ARCH-SCHED-GLOBAL-MEM-REFACTOR)

**核心决策**:
1. 删除 `vllm2024.rs` 中冗余 `LMCacheConfig/LmcacheState/CacheEntry/CacheHit/CacheLevel`。
2. 保留 `ChunkedConfig`，并融合为 Prefill 阶段的页面规划能力（非 Prefill/Decode 混批）。
3. 严格禁止建立任何 `SwiftKV` 或 CPU 端的 KV Cache 泛型蒸馏接口。KV 数据受限于 GPU 端 TurboQuant 双轨位宽压缩锁定，不再允许 CPU 染指任何反量化或浮点重新计算的行为。

**统一能力入口**:
- `plan_prefill(prompt_tokens, chunk_size) -> PrefillPlan`
- `prepare_prefill_with_auto_reuse(request_id, tokens) -> PrefillPlan`
- `allocate_page_in_pipeline(pipeline, tier) -> Result<PhysicalId, Error>`
- `prepare_next_turn(session_id)`（释放 `Working`，保留 `Conversation`）

### 6. JIT 兼容性声明 (ARCH-SCHED-REFACTOR-JIT)

| 组件 | 是否需要新 Kernel | 理由 |
|------|-------------------|------|
| `KvPrefixIndex` | ❌ | 纯索引结构，属于调度层 |
| `SessionKvCache` | ❌ | 会话元数据管理 |
| `KvPipeline` | ❌ | 页表命名空间扩展 |
| `BatchOrderPolicy` | ❌ | 批构建顺序策略 |
| `ChunkedConfig` 融合 | ❌ | 页面规划逻辑，不改变算子实现 |

**结论**: 调度器重构全部位于调度/内存管理层，不涉及 JIT 编译器管线。

### 7. 张量驱动配置系统 (ARCH-LOADER-TENSOR)

> **关联需求**: REQ-LOADER-022, REQ-LOADER-023
> **核心原则**: Ω1 真实性原则 - 物理张量形状是唯一的真理 (Ground Truth)。

#### 7.1 TensorProvider 抽象 (ARCH-LOADER-PROVIDER)

为了屏蔽文件格式差异 (SafeTensors vs GGUF vs ONNX) 并彻底消除动态分发开销，系统定义统一的**静态分发**接口。

**接口定义**:
```rust
pub trait TensorProvider {
    fn tensor_info(&self, name: &str) -> Option<TensorMeta>;
    fn iter_tensors(&self) -> impl Iterator<Item = TensorMeta>; // impl Iterator 确保不可用作 dyn
    fn load_tensor_data(&self, name: &str) -> Result<Cow<'_, [u8]>>;
}
```

**约束**:
- **禁止动态分发 (NO_DYN)**: `iter_tensors` 返回 `impl Iterator`，这在 Rust 中天然使得 Trait 不具备对象安全性 (Not Object Safe)。这强制所有使用处必须通过泛型 `<P: TensorProvider>` 进行**静态单态化 (Monomorphization)**，确保零运行时开销。
- **零拷贝数据**: `load_tensor_data` 返回 `Cow<'_, [u8]>`，对于 `mmap` 的 SafeTensors 直接返回引用，对于 GGUF/ONNX 可能需要临时分配（如果不支持 mmap）。

#### 7.2 角色探测 (ARCH-LOADER-ROLE)

不再依赖硬编码的张量名称，而是基于语义模式匹配。

| 逻辑角色 | 匹配模式 (Regex/Contains) | 示例 |
|---------|-------------------------|------|
| `Embedding` | `embed` \| `wte` \| `word_embeddings` | `model.embed_tokens`, `transformer.wte` |
| `AttnQuery` | `q_proj` \| `query` \| `wq` | `layers.0.self_attn.q_proj` |
| `AttnKey` | `k_proj` \| `key` \| `wk` | `layers.0.self_attn.k_proj` |
| `AttnValue` | `v_proj` \| `value` \| `wv` | `layers.0.self_attn.v_proj` |
| `AttnOutput`| `o_proj` \| `output` \| `wo` | `layers.0.self_attn.o_proj` |
| `FfnGate` | `gate_proj` \| `w1` \| `ffn_gate` | `layers.0.mlp.gate_proj` |
| `FfnUp` | `up_proj` \| `w3` \| `ffn_up` | `layers.0.mlp.up_proj` |
| `FfnDown` | `down_proj` \| `w2` \| `ffn_down` | `layers.0.mlp.down_proj` |

#### 7.3 拓扑推导逻辑 (ARCH-LOADER-TOPOLOGY)

推导必须遵循 **Truth > Hint > Fallback** 优先级。

1.  **Vocab Size**: `Embedding.shape[0]` (或较大维度)。
2.  **Hidden Size**: `Embedding.shape[1]` (或较小维度)。
3.  **Head Dim**:
    - 计算 `Q_out = Q_proj.output_dim`
    - 计算 `K_out = K_proj.output_dim`
    - 遍历候选 `[32, 64, 80, 96, 128, 256]`，找到能同时整除 `Q_out` 和 `K_out` 的值。
    - **歧义处理**: 如果有多个候选值（如 64 和 128 均合法），则**报错** (InvalidConfig)，除非有显式 Metadata Hint。禁止猜测。
4.  **Num Heads**: `Q_out / Head Dim`。
5.  **Num KV Heads**: `K_out / Head Dim`。
6.  **Intermediate Size**: `FfnGate.output_dim` 或 `FfnUp.output_dim`。
7.  **Num Layers**: 解析所有张量名称中的层索引（如 `layers.N`），取最大值 + 1。

#### 7.4 验证 (ARCH-LOADER-VERIFY)

- **物理约束验证**:
  - `Q_out % Head Dim == 0`
  - `K_out % Head Dim == 0`
  - `Num Heads % Num KV Heads == 0` (GQA 约束)
- **Ω1 违规**: 任何推导出的配置与物理张量形状冲突时，以**物理张量**为准，并可能需要覆盖 Config。

---

### 5.3 JIT 图 Shape 动态化 (REQ-JIT-GRAPH-001)

> **关联需求**: REQ-JIT-GRAPH-001
> **实现位置**: `gllm-kernels/src/compiler/graph.rs`（新增 `SymDim`）+ `src/compat/jit_helpers.rs`（更新图构建函数）

#### 问题描述

当前 `CompilerGraph` 在构建时如果将所有张量维度硬编码为具体的 `usize` 数值，那么在 decoder 推理循环中，`total_seq`（KV cache 已缓存的序列长度）每个 decode step 递增 1，会导致需要 shape binding 的图（如 `AttentionConsume`）每步都不同，触发重编译。旧版基于 `compile_decode_jit` 的代码曾存在此类问题。

根据 **ARCH-CPU-GPU-UNIFIED (见 §8)** 规则，CPU 与 GPU 的编译管线在 JIT 缓存策略上必须完全统一，并严格遵守 `jit-cache-protocol.md` 定义的模型级缓存原则。


#### 设计方案：SymDim

在 `gllm-kernels/src/compiler/graph.rs` 中新增 `SymDim` 枚举，替代裸 `usize` 作为张量维度类型：

```rust
/// 张量维度：可以是具体值，也可以是符号名（运行时绑定）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymDim {
    /// 编译时已知的具体维度
    Concrete(usize),
    /// 运行时绑定的符号维度，名称用于 shape binding 查找
    Symbolic(String),
}
```

`CompilerGraph::add_tensor` 的 shape 参数类型从 `&[usize]` 改为 `&[SymDim]`，允许混合具体维度和符号维度：

```rust
// 示例：hidden_size 是具体值，total_seq 是符号维度
graph.add_tensor("kv_cache", &[
    SymDim::Concrete(num_layers),
    SymDim::Symbolic("total_seq".to_string()),
    SymDim::Concrete(kv_dim),
]);
```

`CompiledLayer::execute` 接受运行时 shape binding，在执行前将符号维度替换为具体值：

```rust
pub struct ShapeBinding {
    pub bindings: HashMap<String, usize>,
}

impl CompiledLayer {
    pub fn execute(&self, inputs: &[*const u8], outputs: &[*mut u8],
                   binding: &ShapeBinding) -> Result<(), KernelError>;
}
```

#### 铁律

- ❌ 禁止在图构建时将 `total_seq`、`batch_size` 等动态维度硬编码为具体数值
- ❌ 禁止因 shape 变化而重新调用 `compile_and_run` 构建新图
- ✅ 动态维度必须声明为 `SymDim::Symbolic`，名称与 `ShapeBinding` 的 key 一致
- ✅ 编译一次，通过 `ShapeBinding` 在每个 decode step 传入当前 `total_seq`
- ✅ `Concrete` 维度在编译时参与代码生成优化（循环展开、tile 大小等）；`Symbolic` 维度在运行时从 binding 读取，生成通用循环

#### 受影响的图构建函数

| 函数 | 文件 | 需要符号化的维度 |
|------|------|----------------|
| `compile_decode_jit` (CachedGQA 图) | `src/compat/jit_helpers.rs` | `total_seq` |
| `compile_decode_jit` (RmsNorm+Q+RoPE 图) | `src/compat/jit_helpers.rs` | `seq_len`（prefill 时动态） |


---

## §8 Zero-Overhead Unified Attention Architecture (ARCH-ATTN-UNIFIED)

> **状态**: 设计完成，待实现
> **关联**: Accuracy First (§0), Zero-Copy (Layer 4), JIT 编译管线 (§5)
> **动机**: GPU prefill 与 CPU/decode attention 当前使用不同 op（`MultiHeadAttention` vs `CachedGQA`），导致数值路径分叉、KV 布局重排冗余、测试复杂度倍增。

### 8.1 核心原则：统一 Contract，不统一 Payload

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ARCH-ATTN-UNIFIED 铁律                                                │
│                                                                         │
│  ✅ 统一语义契约（AttentionSemantics）                                  │
│  ✅ 统一访问协议（KvView, WeightView, PositionContract）               │
│  ✅ 保留原始物理载荷（零拷贝、零重排、零 dtype 膨胀）                   │
│  ✅ JIT 直接面向 View/Contract codegen                                  │
│                                                                         │
│  ❌ 禁止为统一而引入公共物理中间格式                                    │
│  ❌ 禁止 KV head-major ↔ seq-major 全量重排作为主路径                   │
│  ❌ 禁止 BF16/F16/quant cache 全量膨胀为 F32 常驻格式                   │
│  ❌ 禁止 paged KV 先拼成 dense K/V 再计算                               │
│  ❌ 禁止 GGUF/ONNX/SafeTensors 权重转成统一 payload                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 五个核心 Contract

#### 8.2.1 `AttentionSemantics`

定义 attention 的数学语义，prefill 与 decode 共享。

| 字段 | 类型 | 说明 |
|------|------|------|
| `mask_mode` | `Causal \| Bidirectional` | 因果/双向 |
| `head_mode` | `MHA \| GQA { ratio } \| MQA` | 头映射 |
| `scaling` | `1/sqrt(head_dim)` | 缩放规则 |
| `rope` | `Option<RoPEConfig>` | 旋转位置编码配置 |
| `visibility` | `Prefill { seq_len } \| Decode { total_seq }` | 可见范围 |

#### 8.2.2 `KvLayoutContract`

描述 KV 的物理布局，不要求转换。

| 字段 | 类型 | 说明 |
|------|------|------|
| `storage_kind` | `DenseSeqMajor \| HeadMajorPersistent \| PagedHeadMajor` | 存储类型 |
| `kv_split` | `SplitHalf \| Interleaved` | K/V 分离方式 |
| `layer_stride` | `usize` | 层间步长 |
| `head_stride` | `usize` | 头间步长 |
| `token_stride` | `usize` | token 间步长 |
| `turbo_quant` | `TurboQuantBits` | 位宽约束 |
| `append_semantics` | `AppendOnly \| Overwrite` | 追加语义 |

#### 8.2.3 `KvView`

在 `KvLayoutContract` 之上的具体逻辑视图。

| 变体 | 说明 | 使用场景 |
|------|------|---------|
| `DenseLocal { k_ptr, v_ptr, seq_len, kv_dim }` | 当前层临时 K/V | Prefill attention |
| `PersistentCache { base_ptr, contract, layer, visible_range }` | 持久 KV cache | Decode attention |
| `PagedCache { page_table, contract, layer, visible_range }` | 分页 KV cache | Paged decode |
| `CompositeView { cached, append }` | cache + 新 token 组合 | Prefill-to-decode 过渡 |

**铁律**: attention kernel 的 K/V 输入必须是 `KvView`，禁止要求调用方先 materialize 成 dense tensor。

#### 8.2.4 `PositionContract`

统一位置来源，支持 RoPE on-the-fly lowering。

| 变体 | 说明 |
|------|------|
| `ContiguousRange { start, len }` | 连续位置（prefill 常见） |
| `ExplicitArray { ptr, len }` | 显式位置数组（decode / 非连续） |
| `OffsetStride { offset, stride }` | 等间距位置 |

**目标**: RoPE 不再要求先 materialize `q_rope/k_rope` 中间张量，允许 lower 到消费时寄存器内变换。

#### 8.2.5 `WeightView`

统一 GGUF/ONNX/SafeTensors 权重访问，保留原始物理载荷。

| 字段 | 类型 | 说明 |
|------|------|------|
| `logical_shape` | `Vec<usize>` | 逻辑形状 |
| `turbo_quant` | `TurboQuantBits` | 极化压缩位宽 |
| `quant_scheme` | `Option<QuantScheme>` | 量化方案（GGUF block quant 等） |
| `base_ptr` | `*const u8` | 数据指针（mmap/device） |
| `stride` | `Vec<usize>` | 步长 |
| `packing` | `PackingDescriptor` | 打包描述（block size, scale/zero layout） |

**铁律**: JIT 直接根据 `WeightView` 生成加载逻辑，禁止要求先转成 dense F32。

### 8.3 Attention 统一拆分：Projection + Consume

不再使用分裂的 `MultiHeadAttention` / `CachedGQA` 两个 op。

统一为两个语义操作：

#### A. `AttentionProjection`

```
input[seq, hidden] + Wq/Wk/Wv + PositionContract
    → QView[seq, q_dim]
    → KEmit[seq, kv_dim]
    → VEmit[seq, kv_dim]
    → KvAppendIntent
```

- 可选融合 RmsNorm
- 可选融合 RoPE（on-the-fly，不写回中间张量）
- `KvAppendIntent` 是副作用声明，不是强制 materialization

#### B. `AttentionConsume`

```
QView + KvView + AttentionSemantics
    → attn_out[seq, q_dim]
```

- 输入是 `KvView`，不是 dense K/V tensor
- JIT 根据 `KvView` 变体生成不同访存逻辑

### 8.4 Prefill / Decode 双 Lowering

共享同一 `AttentionSemantics`，lower 到不同执行计划：

```
                    AttentionSemantics
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
      Prefill Lowering         Decode Lowering
              │                       │
    ┌─────────┴─────────┐   ┌────────┴────────┐
    │ DenseLocal KvView │   │ Persistent/Paged │
    │ Full-seq causal   │   │ KvView           │
    │ High throughput   │   │ Cached attention  │
    │ No cache read     │   │ Low latency      │
    └───────────────────┘   └─────────────────┘
```

#### Prefill 路径（零额外 KV round-trip）

1. `AttentionProjection` → Q, K, V（临时 buffer）
2. `AttentionConsume(Q, DenseLocal{K,V})` → attn_out
3. **并行副作用**: `KvAppend(K, V → persistent cache)`
4. 后续 FFN

**关键**: attention 直接消费临时 K/V，不经过 cache round-trip。

#### Decode 路径（零 KV 重排）

1. `AttentionProjection` → Q, K_new, V_new
2. `KvAppend(K_new, V_new → persistent cache)`
3. `AttentionConsume(Q, PersistentKvView{cache})` → attn_out
4. 后续 FFN

**关键**: attention 直接从 persistent cache 原位读取，不重排为 seq-major dense tensor。

### 8.5 JIT + KV 融合边界

#### 融合进 JIT 的（KV Access Lowering）

| 融合项 | 说明 |
|--------|------|
| stride addressing | 根据 `KvLayoutContract` 生成 head/token/layer 索引 |
| page lookup | 根据 `PagedKvView` 生成 page table 查找 |
| append window | 将 append region 合并进可见窗口 |
| static layout dispatch | 根据 TurboQuantBits 生成载入指令 |
| RoPE on-the-fly | 根据 `PositionContract` 在消费时做旋转 |

#### 不融合进 JIT 的（保留在 Engine/Scheduler 层）

| 不融合项 | 归属层 |
|----------|--------|
| page allocation / free | MemoryManager |
| swap / eviction | Scheduler |
| prefix dedup | KvPrefixIndex |
| request lifecycle | Executor |
| scheduling policy | PolicyEngine |

### 8.6 No Redundant KV Reformat (ARCH-ATTN-NO-REFORMAT)

**铁律**: persistent KV cache 不得在主路径中被全量重排为 dense seq-major 形式后再做 attention。

已修复路径 (2026-03-22)：

| 路径 | 原违规操作 | 修复结果 |
|------|---------|---------|
| GPU decode `jit_cached_attention` | 下载全量 KV → CPU CachedGQA | ✅ 已改为 GPU native cached attention |
| GPU decode `download_kv_cache_to_host` | 每层下载 ~180MB KV 到 CPU | ✅ 已删除，attention 直接读 GPU KV cache |

### 8.7 实现路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| P0 | 定义 5 个 core contract 类型 | ✅ 已完成 |
| P0 | GPU prefill: 拆分 projection + consume，消除 QKV fusion 问题 | ✅ 已完成 |
| P1 | GPU decode: native cached attention kernel（消除 CPU round-trip） | ✅ 已完成 |
| P1 | KvView lowering: JIT 直接消费 head-major cache | ✅ 已完成 |
| P2 | RoPE on-the-fly: 消除 q_rope/k_rope 中间张量 | ✅ 已完成 |
| P2 | WeightView: GGUF/SafeTensors/ONNX 统一 view | ✅ 已完成 |
| P3 | PagedKvView: paged attention 原位访问 | ✅ 已完成 (2026-03-22) |

### 8.8 (架构级物理废弃) DType 全链路自适应 (ARCH-DTYPE-ADAPTIVE)

> 🚨 **架构合规性警报 (Architect Veto)**:
> 曾经规划的 `DType全链路自适应` 以及动态计算 `dtype_size` 以支持在 JIT 管线中随意游走 F16/BF16/F32 的架构，现已被**全盘否决并物理销毁**。
>
> 1. 原方案寄望于在运行时通过 `computation_dtype_from_config` 等函数获取浮点类型的字节宽度，并据此分派汇编宏。
> 2. 在全系统的底盘升构为 **TurboQuant** 之后，系统进入了“静态装配期强行降阶（Load-Time Annihilation）”。
> 3. Mega-Kernel 强制消除多态加载，所有张量强转至极化后的定点/微浮点硬派算子规格。不再存在动态检测 `F16`、`BF16` 并相应生成不同载入指令的软路由。
>
> **因此，原本长达两百行的关于 `dtype.size_bytes()` 动态寻址的复杂映射表与实现方案被永久废除。禁止后继 Agent 实现此模块中的硬编码探测或多态汇编下发逻辑！**

---



---

## §8 Zero-Overhead Unified Attention Architecture (ARCH-ATTN-UNIFIED)

> **状态**: 设计完成，待实现
> **关联**: Accuracy First (§0), Zero-Copy (Layer 4), JIT 编译管线 (§5)
> **动机**: GPU prefill 与 CPU/decode attention 当前使用不同 op（`MultiHeadAttention` vs `CachedGQA`），导致数值路径分叉、KV 布局重排冗余、测试复杂度倍增。

### 8.1 核心原则：统一 Contract，不统一 Payload

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ARCH-ATTN-UNIFIED 铁律                                                │
│                                                                         │
│  ✅ 统一语义契约（AttentionSemantics）                                  │
│  ✅ 统一访问协议（KvView, WeightView, PositionContract）               │
│  ✅ 保留原始物理载荷（零拷贝、零重排、零 dtype 膨胀）                   │
│  ✅ JIT 直接面向 View/Contract codegen                                  │
│                                                                         │
│  ❌ 禁止为统一而引入公共物理中间格式                                    │
│  ❌ 禁止 KV head-major ↔ seq-major 全量重排作为主路径                   │
│  ❌ 禁止 BF16/F16/quant cache 全量膨胀为 F32 常驻格式                   │
│  ❌ 禁止 paged KV 先拼成 dense K/V 再计算                               │
│  ❌ 禁止 GGUF/ONNX/SafeTensors 权重转成统一 payload                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 五个核心 Contract

#### 8.2.1 `AttentionSemantics`

定义 attention 的数学语义，prefill 与 decode 共享。

| 字段 | 类型 | 说明 |
|------|------|------|
| `mask_mode` | `Causal \| Bidirectional` | 因果/双向 |
| `head_mode` | `MHA \| GQA { ratio } \| MQA` | 头映射 |
| `scaling` | `1/sqrt(head_dim)` | 缩放规则 |
| `rope` | `Option<RoPEConfig>` | 旋转位置编码配置 |
| `visibility` | `Prefill { seq_len } \| Decode { total_seq }` | 可见范围 |

#### 8.2.2 `KvLayoutContract`

描述 KV 的物理布局，不要求转换。

| 字段 | 类型 | 说明 |
|------|------|------|
| `storage_kind` | `DenseSeqMajor \| HeadMajorPersistent \| PagedHeadMajor` | 存储类型 |
| `kv_split` | `SplitHalf \| Interleaved` | K/V 分离方式 |
| `layer_stride` | `usize` | 层间步长 |
| `head_stride` | `usize` | 头间步长 |
| `token_stride` | `usize` | token 间步长 |
| `turbo_quant` | `TurboQuantBits` | 位宽约束 |
| `append_semantics` | `AppendOnly \| Overwrite` | 追加语义 |

#### 8.2.3 `KvView`

在 `KvLayoutContract` 之上的具体逻辑视图。

| 变体 | 说明 | 使用场景 |
|------|------|---------|
| `DenseLocal { k_ptr, v_ptr, seq_len, kv_dim }` | 当前层临时 K/V | Prefill attention |
| `PersistentCache { base_ptr, contract, layer, visible_range }` | 持久 KV cache | Decode attention |
| `PagedCache { page_table, contract, layer, visible_range }` | 分页 KV cache | Paged decode |
| `CompositeView { cached, append }` | cache + 新 token 组合 | Prefill-to-decode 过渡 |

**铁律**: attention kernel 的 K/V 输入必须是 `KvView`，禁止要求调用方先 materialize 成 dense tensor。

#### 8.2.4 `PositionContract`

统一位置来源，支持 RoPE on-the-fly lowering。

| 变体 | 说明 |
|------|------|
| `ContiguousRange { start, len }` | 连续位置（prefill 常见） |
| `ExplicitArray { ptr, len }` | 显式位置数组（decode / 非连续） |
| `OffsetStride { offset, stride }` | 等间距位置 |

**目标**: RoPE 不再要求先 materialize `q_rope/k_rope` 中间张量，允许 lower 到消费时寄存器内变换。

#### 8.2.5 `WeightView`

统一 GGUF/ONNX/SafeTensors 权重访问，保留原始物理载荷。

| 字段 | 类型 | 说明 |
|------|------|------|
| `logical_shape` | `Vec<usize>` | 逻辑形状 |
| `turbo_quant` | `TurboQuantBits` | 极化压缩位宽 |
| `quant_scheme` | `Option<QuantScheme>` | 量化方案（GGUF block quant 等） |
| `base_ptr` | `*const u8` | 数据指针（mmap/device） |
| `stride` | `Vec<usize>` | 步长 |
| `packing` | `PackingDescriptor` | 打包描述（block size, scale/zero layout） |

**铁律**: JIT 直接根据 `WeightView` 生成加载逻辑，禁止要求先转成 dense F32。

### 8.3 Attention 统一拆分：Projection + Consume

不再使用分裂的 `MultiHeadAttention` / `CachedGQA` 两个 op。

统一为两个语义操作：

#### A. `AttentionProjection`

```
input[seq, hidden] + Wq/Wk/Wv + PositionContract
    → QView[seq, q_dim]
    → KEmit[seq, kv_dim]
    → VEmit[seq, kv_dim]
    → KvAppendIntent
```

- 可选融合 RmsNorm
- 可选融合 RoPE（on-the-fly，不写回中间张量）
- `KvAppendIntent` 是副作用声明，不是强制 materialization

#### B. `AttentionConsume`

```
QView + KvView + AttentionSemantics
    → attn_out[seq, q_dim]
```

- 输入是 `KvView`，不是 dense K/V tensor
- JIT 根据 `KvView` 变体生成不同访存逻辑

### 8.4 Prefill / Decode 双 Lowering

共享同一 `AttentionSemantics`，lower 到不同执行计划：

```
                    AttentionSemantics
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
      Prefill Lowering         Decode Lowering
              │                       │
    ┌─────────┴─────────┐   ┌────────┴────────┐
    │ DenseLocal KvView │   │ Persistent/Paged │
    │ Full-seq causal   │   │ KvView           │
    │ High throughput   │   │ Cached attention  │
    │ No cache read     │   │ Low latency      │
    └───────────────────┘   └─────────────────┘
```

#### Prefill 路径（零额外 KV round-trip）

1. `AttentionProjection` → Q, K, V（临时 buffer）
2. `AttentionConsume(Q, DenseLocal{K,V})` → attn_out
3. **并行副作用**: `KvAppend(K, V → persistent cache)`
4. 后续 FFN

**关键**: attention 直接消费临时 K/V，不经过 cache round-trip。

#### Decode 路径（零 KV 重排）

1. `AttentionProjection` → Q, K_new, V_new
2. `KvAppend(K_new, V_new → persistent cache)`
3. `AttentionConsume(Q, PersistentKvView{cache})` → attn_out
4. 后续 FFN

**关键**: attention 直接从 persistent cache 原位读取，不重排为 seq-major dense tensor。

### 8.5 JIT + KV 融合边界

#### 融合进 JIT 的（KV Access Lowering）

| 融合项 | 说明 |
|--------|------|
| stride addressing | 根据 `KvLayoutContract` 生成 head/token/layer 索引 |
| page lookup | 根据 `PagedKvView` 生成 page table 查找 |
| append window | 将 append region 合并进可见窗口 |
| static layout dispatch | 根据 TurboQuantBits 生成载入指令 |
| RoPE on-the-fly | 根据 `PositionContract` 在消费时做旋转 |

#### 不融合进 JIT 的（保留在 Engine/Scheduler 层）

| 不融合项 | 归属层 |
|----------|--------|
| page allocation / free | MemoryManager |
| swap / eviction | Scheduler |
| prefix dedup | KvPrefixIndex |
| request lifecycle | Executor |
| scheduling policy | PolicyEngine |

### 8.6 No Redundant KV Reformat (ARCH-ATTN-NO-REFORMAT)

**铁律**: persistent KV cache 不得在主路径中被全量重排为 dense seq-major 形式后再做 attention。

已修复路径 (2026-03-22)：

| 路径 | 原违规操作 | 修复结果 |
|------|---------|---------|
| GPU decode `jit_cached_attention` | 下载全量 KV → CPU CachedGQA | ✅ 已改为 GPU native cached attention |
| GPU decode `download_kv_cache_to_host` | 每层下载 ~180MB KV 到 CPU | ✅ 已删除，attention 直接读 GPU KV cache |

### 8.7 实现路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| P0 | 定义 5 个 core contract 类型 | ✅ 已完成 |
| P0 | GPU prefill: 拆分 projection + consume，消除 QKV fusion 问题 | ✅ 已完成 |
| P1 | GPU decode: native cached attention kernel（消除 CPU round-trip） | ✅ 已完成 |
| P1 | KvView lowering: JIT 直接消费 head-major cache | ✅ 已完成 |
| P2 | RoPE on-the-fly: 消除 q_rope/k_rope 中间张量 | ✅ 已完成 |
| P2 | WeightView: GGUF/SafeTensors/ONNX 统一 view | ✅ 已完成 |
| P3 | PagedKvView: paged attention 原位访问 | ✅ 已完成 (2026-03-22) |

### 8.9 (架构变迁) TurboQuant 静态阻塞规约 (ARCH-TQ-BLOCKING)

> **关联**: TurboQuant 架构, Zero-Overhead Memory Pool
> **状态**: 已替代旧有的 DType-Aware 性能优化。

由于系统全面废除了多态的 `DType` 运行时分支，所有的 GEMM 阻塞 (Blocking) 逻辑（KC/MC/NC 计算）现在完全基于 **硬件检测 (hw_constraints.rs)** 和 **TurboQuant 固定位宽**。

`device_profile.rs:gemm_blocking()` 函数已退役。取而代之的是 `TurboQuant` 为 W4A4 和 W8A8 场景直接提供的硬连线 (hardwired) 最佳参数：
- 不再包含关于 F16/BF16/F32 的 `elem_bytes` 动态除法
- 启发式缓存调优 (WisdomDb) 仅使用 `QuantConstraint` 作为主键查询

---

### 8.9 GPU 数据通路优化 (ARCH-GPU-DATAPATH)

> **状态**: 设计完成，待实现
> **关联**: Zero-Copy (Layer 4), ARCH-ATTN-UNIFIED (§8), ARCH-DTYPE-ADAPTIVE (§8.8)
> **动机**: GPU 推理路径中存在冗余数据搬运（逐 token DtoD、每 step 权重 htod、Metal 中间 buffer），消除后可显著降低 kernel launch 开销和 PCIe/内存带宽占用。

#### 8.9.1 核心原则

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ARCH-GPU-DATAPATH 铁律                                                │
│                                                                         │
│  ✅ KV cache 写入必须通过 GPU kernel scatter（一次 launch）             │
│  ✅ 权重首次上传后常驻 GPU，后续 step 直接 DtoD 复制                   │
│  ✅ Metal shared memory 路径直接指针写入，禁止中间 buffer               │
│  ✅ PagedKvView 三后端（CUDA/HIP/Metal）统一支持                       │
│                                                                         │
│  ❌ 禁止逐 token 逐 head 发起独立 DtoD API 调用                       │
│  ❌ 禁止每 step 重复 htod 上传不变的权重数据                           │
│  ❌ 禁止 Metal 路径分配中间 host buffer 做 dtoh→htod 转发              │
│  ❌ 禁止 paged attention 仅支持单一 GPU 后端                           │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.9.2 KV Cache Scatter Kernel (REQ-ARCH-004)

**问题**: `gpu_write_kv_cache_device` 对 GQA 多头模型发起 O(num_kv_heads × seq_len) 次独立 `cuMemcpyDtoD_v2` 调用。8 heads × 512 seq = 4096 次 API 调用，kernel launch overhead 远超实际拷贝时间。

**设计**: 新增 `OpKind::KvScatterWrite` JIT kernel，一次 launch 完成所有 head 的 KV 散写。

```
源布局: [seq_len, kv_dim] (interleaved heads)
目标布局: per-head [max_seq, head_dim] within KV cache

Kernel ABI:
  inputs:  (k_src_ptr, v_src_ptr, kv_cache_ptr, scatter_meta_ptr)
  params:  (seq_len, num_kv_heads, head_dim, kv_dim, write_start,
            layer_offset, half_offset, head_stride, byte_width)
  grid:    (num_kv_heads, seq_len, 1)
  block:   (min(head_dim, max_threads_per_block), 1, 1)

每个 thread block 处理一个 (head, token) 对:
  src_off = (blockIdx.y * kv_dim + blockIdx.x * head_dim) * byte_width
  dst_k = kv_cache_ptr + layer_offset + blockIdx.x * head_stride
        + (write_start + blockIdx.y) * head_dim * byte_width
  dst_v = dst_k + half_offset
  memcpy(dst_k + threadIdx.x * byte_width,
         k_src + src_off + threadIdx.x * byte_width, byte_width)
  memcpy(dst_v + threadIdx.x * byte_width,
         v_src + src_off + threadIdx.x * byte_width, byte_width)
```

**实现路径**:
1. gllm-kernels: 新增 `OpKind::KvScatterWrite` + PTX/HIP/MSL codegen
2. gllm: `gpu_compile.rs` 编译 scatter kernel + 替换逐 token DtoD 调用
3. MQA 快速路径保留（单头时 2 次 DtoD 仍优于 kernel launch）

#### 8.9.3 GPU 权重常驻缓存 (REQ-ARCH-005)

**问题**: 每 decode step 的层循环内，9 个权重 tensor 通过 htod 上传到 GPU（30 层 × 9 权重 = 270 次 htod/step）。权重在推理期间不变。

**设计**: `GpuWeightCache` — 首次 forward 时一次性上传所有层权重到 GPU 并缓存。

```
struct GpuWeightCache {
    /// 每层的权重 GPU buffer: layer_idx → (tensor_name → device_ptr)
    layers: Vec<HashMap<String, u64>>,
    /// 总显存占用（字节）
    total_bytes: usize,
    /// 是否已初始化
    initialized: bool,
}

生命周期:
  1. 首次 forward: load_all_weights → htod → 缓存 device_ptr
  2. 后续 forward: 层循环内 DtoD 从缓存复制到 kernel input buffer
  3. Executor drop: 释放所有 GPU buffer

显存预算控制:
  - 小模型 (< 2GB): 全量缓存
  - 中模型 (2-8GB): 全量缓存（如果可用显存 > 模型大小 × 2）
  - 大模型 (> 8GB): 按需加载（保持当前 htod 路径作为 fallback）
```

**实现路径**:
1. gllm: `GpuWeightCache` 结构体 + 初始化/查询/释放 API
2. gllm: `gpu_compile.rs` 各后端 forward 函数检测缓存 → DtoD 或 htod fallback
3. gllm: `Executor` 持有 `GpuWeightCache`，drop 时释放

#### 8.9.4 Metal Prefill KV 直写 (REQ-ARCH-006)

**问题**: Metal prefill 路径 KV write 走 dtoh → 中间 buffer → `gpu_write_kv_cache`(htod)。Metal 使用 shared memory（`[buffer contents]` 返回 CPU 可见指针），dtoh/htod 实际是 memcpy，但中间 buffer 分配和拷贝仍是不必要的开销。

**设计**: 直接通过 shared memory 指针写入 KV cache，消除中间 buffer。

```
当前路径 (3 步):
  1. dtoh: GPU K/V buffer → k_host/v_host Vec<u8>
  2. repack: k_host → per-head k_packed
  3. htod: k_packed → KV cache

优化路径 (1 步):
  1. 直接指针写入: K/V buffer.contents() → KV cache.contents()
     (两者都是 CPU 可见的 shared memory 指针)

实现:
  fn metal_write_kv_direct(
      k_buf: &MetalBuffer,    // projection 输出
      v_buf: &MetalBuffer,
      kv_cache_handle: &KvCacheHandle,  // KV cache (shared memory)
      ...
  ) {
      let k_src = k_buf.as_device_ptr() as *const u8;
      let kv_dst = kv_cache_handle.0 as *mut u8;
      // 直接 per-head memcpy，无中间 buffer
      for head in 0..num_kv_heads {
          for s in 0..seq_len {
              ptr::copy_nonoverlapping(k_src + src_off, kv_dst + dst_off, head_dim * byte_width);
          }
      }
  }
```

#### 8.9.5 PagedKvView 三后端统一 (REQ-ARCH-007)

**问题**: PagedKvView (P3) 仅接入 CUDA decode 路径。HIP/Metal decode 路径仍使用 dense KV cache。

**设计**: 将 CUDA paged attention 路径复制到 HIP/Metal，适配各后端 API。

```
CUDA (已完成):
  build_gpu_paged_attention_graph → AttentionStrategy::Paged
  gpu_write_kv_paged → DtoD scatter into physical pages
  gpu_alloc_paged_kv_cache → cuMemAlloc
  gpu_upload_page_table → cuMemcpyHtoD

HIP (待实现):
  复用 build_gpu_paged_attention_graph (共享)
  gpu_write_kv_paged_hip → hipMemcpyDtoD scatter
  gpu_alloc_paged_kv_cache_hip → hipMalloc
  gpu_upload_page_table_hip → hipMemcpyHtoD

Metal (待实现):
  复用 build_gpu_paged_attention_graph (共享)
  metal_write_kv_paged → shared memory 直接写入
  metal_alloc_paged_kv_cache → MTLDevice.newBuffer
  metal_upload_page_table → MTLBuffer.contents() memcpy
```

#### 8.9.6 依赖关系

```
REQ-ARCH-004 (KV Scatter) ← 无依赖，可立即开始
REQ-ARCH-005 (权重缓存)   ← 无依赖，可与 004 并行
REQ-ARCH-006 (Metal 直写) ← 无依赖，可并行
REQ-ARCH-007 (Paged 三端) ← 依赖 004（paged 路径也需要 scatter kernel）
```
