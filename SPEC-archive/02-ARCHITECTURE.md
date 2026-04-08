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
| `bits` | `u8` | ✅ | 权重物理精度 (如 4 或 8) |
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
│  量化: Backend trait 仅提供 quantized_matmul (极化相乘) 默认约束，取消孤立解包 │
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
    - **DType Adapter**: **(已弃用)** Loader 层不再根据硬件进行动态类型分发。

6.  **权重加载与 TurboQuant 运行时优化 (ARCH-LOADER-TURBO-QUANT)**
    - **职责边界**: gllm 是纯推理引擎，支持加载 SafeTensors/GGUF/ONNX 全格式权重文件。Loader 负责读取权重文件、解包张量、提取元数据，并将结果交给 JIT 执行引擎。
    - **QuantType 驱动 JIT**: 加载时检测到的 QuantType 直接驱动 JIT 生成对应的硬件原生内核。推理过程中无类型判断分支，无 Amax 动态探测。
    - **TurboQuant 运行时优化**: gllm 在前向传播中执行在线 FWHT 旋转、KV 非对称量化、RaBitQ 无偏修正等数学优化，使推理精度逼近无损。详见 §11。
    - **Backend::dequantize 废除**: 不再提供反量化能力。JIT 内核直接读取定点网格点，数学运算基于整数/微字节累加器进行，零分支执行。

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
    → 移交至 Executor 队列
    → Scheduler 批量装载 (Continuous Batching)
    → GraphExecutor 触名单体巨型计算图 (执行 Mega-Kernel)
    → 物理硬件进行 [Embedding → (Layer N × RmsNorm/QKV/Attn/FFN) → Pooling] 等端到端计算
    → Tokenizer 解码 / Tensor 取回
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
├── backend/            # 后端检测与越界截断 (OOM Halt)
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

> **实现状态**: ✅ IMPLEMENTED
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
| `quant_type` | `QuantType` | 量化格式 |
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
| `quant_type` | `QuantType` | 量化格式 |
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
| static layout dispatch | 根据 QuantType 生成载入指令 |
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



---
### 8.9 (架构变迁) TurboQuant 静态阻塞规约 (ARCH-TQ-BLOCKING)

> **关联**: TurboQuant 架构, Zero-Overhead Memory Pool
> **状态**: 已替代旧有的 DType-Aware 性能优化。

由于系统全面废除了多态的 `DType` 运行时分支，所有的 GEMM 阻塞 (Blocking) 逻辑（KC/MC/NC 计算）现在完全基于 **硬件检测 (hw_constraints.rs)** 和 **TurboQuant 特化精度**。

`device_profile.rs:gemm_blocking()` 函数已退役。取而代之的是 `TurboQuant` 为 W4A4 和 W8A8 场景直接提供的硬连线 (hardwired) 最佳参数：
- 不再包含关于 F16/BF16/F32 的 `elem_bytes` 动态除法
- 启发式缓存调优 (WisdomDb) 仅使用 `QuantConstraint` 作为主键查询

---

### 8.10 GPU 数据通路优化 (ARCH-GPU-DATAPATH)

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

---

## §9 大一统 JIT 底层物理架构 (ARCH-MEGA-KERNEL)

> **实现状态**: ✅ IMPLEMENTED — request_state.rs(654L/10 tests), routing.rs(467L/7 tests), sensors.rs(226L/3 tests)，作为所有后端执行器的最终约束（SSOT）。
> **核心哲学**: JIT 编译只在模型加载与 Autotuning 期发生。推理热路径绝无任何编译机制。

### 9.1 Mega-Kernel 块级路由 (In-Kernel Dispatch)

**问题**: 应对大并发下的请求形态分歧（如稀疏/稠密差异），传统引擎依赖控制流引发分支预测惩罚或多次 Kernel Launch。

**铁律约束**:
- **仅发射单一内核**: 每一轮 Decode 或 Chunked Prefill，全系统 **仅 Launch 唯一一个 Mega-Kernel**。
- **取消主机条件网**: 禁止在主机侧 (CPU Host) 为 `Gate-First-Skip` 等建立多线程路调度。
- **块内联路**: SM 核心内的 Thread Block 直接读取 `Request_State_Table`。条件不满足时，不破坏控制流掩码，直接在 `Shared Memory` / 寄存器堆通过向量外设掩码（AVX512 `vcompress` / GPU `Prefix Sum`）执行**物理挤压聚拢 (Ragged Tensor Compaction)**。
- **Compact→Execute→Scatter 三段式循环**: 挤压聚拢仅仅是第一步（Compact）。在没有 Padding 气泡的连续稠密矩阵中执行完核函数运算后（Execute），必须**按原始 Request 偏移进行原位散射回写（Scatter）**，还原到初始 Batch 位置。整个 Compact→Execute→Scatter 流程在单次 Kernel Launch 内闭环。
- **值域隔离的分组防线 (Range-Aware Compact Grouping)**: 针对极低精度（如 W4A4）下的跨请求数值污染风险，Compact 过程**严禁单纯按 Batch 顺序挤压**。必须利用 §9.5 尾段观测白嫖到的 `Entropy` 和 `Residual Delta` 指标，在挤压时将激活值域（Activation Range）相近的请求聚集在同一 GEMM Tile 内。值域悬殊的请求互相物理隔离，防止小信号被大信号的量化噪声静默吞噬，确保 4-bit 有效精度刃尽其用。

### 9.2 全域热修补 (Global Consensus Hot JMP Patching)

**问题**: 针对无法通过掩码消除的系统级静滞期（如死寂的冷板凳 MoE 专家，极长的系统前缀树）。

**铁律约束**:
- **JIT Director Daemon（后台长臂监控器）**: 后台常驻纯 Rust `JIT Director Daemon` 线程。定期无损挂载扫描全体 KV Page Headers（被 §9.5 Epilogue 白嫖写入的 telemetry 数据）。利用**半衰期积分池（Decaying Reservoir）**平滑指标变异，配合**卡尔曼平滑器（Kalman Limiters）**在数十万 Token 级别的长尾维度上判定是否达成全局共识不可逆突变（如注意力持续数百个 Batch 趋近静默态、冷专家百万 Token 零命中等）。
- **Trampolines 占位符**: 运行期代码块固定留有 `NOP Slide` / `jmp` 占位，分支预测开销为 0。
- **原子覆盖**: 捕获全域共识不可逆突变后，后台在沙盒完成验证并计算新指令绝对偏移量。主系统调用原子写操作（Atomic Overwrite），将 `.text` 执行内存段上的 5-bytes 长 `jmp` 立刻热覆盖到新入口，瞬时重构拓扑图（Graph Collapsing/DCE）。整个推理流水线在毫无波澜的一瞬间完成全局拓扑坍缩重构。

### 9.3 残差数据总线物理结构 (The Residual Bus Injection)

**定论**: 残差流 `x_out = x_in + Layer(x_in)` 必须被编译器重构为一条开放的 **插入端口 (Injection Port)** 与 **召回端口 (Recall Port)**，彻底放开 Transformer 第一性原理。

- **晚期知识融合 (Late-Fusion RAG)**: 外部大体量检索向量可通过指定的锚点直接使用极速的 `Vector Add` 指令汇编 `LDG.E` 植入到特定语义深处，拒绝自前置 Embedding 的无效算力爬行。
- **投机截断与通用降维 (Early-Exit & Intent NLU)**: 
  - 允许在中间层挂载微型 `lm_head`。概率逼近阈值时直接抛出拦截信标截断计算。
  - 支持 `Pure_Decode` API 模式运行。模型仅提取语义核心区残差特征，化身为超高速多意图识别神塔。
- **零延迟飞行护栏 (In-Flight Guardrail)**: 利用附加极简线性分类探针（Micro Probe）寄生挂载，进行安全护栏瞬时探测。一旦探测到风险特征越过 `HaltAndVeto` 阈值，Mega-Kernel 在生成下一个 Token 前当场物理切断（Amputate) 熔断管线。

### 9.4 纯逻辑专家解绑与零卡顿置换 (Extreme MoE Disaggregation)

**铁律约束**:
- **核内分发与零启动开销**: 绝不去启动子 Kernel。Thread Block 利用内置字典读取 Gate 路由，汇编内部 `jmp` 直接跃迁到对应专家的权重读取区。
- **异步预取掩蔽**: 基于 TurboQuant（4-bit 以下压制），系统利用 `cuMemPrefetchAsync` 将低频/离线专家的高速预取时间通过计算流水完美掩蔽。
- **去优化退回陷阱 (De-optimization Bailout)**: 针对被热修补判定剔除的死寂专家，插入“去优化处理惩罚极 (Uncommon Trap)”。若突遭访问，触发错误的 Thread Block 仅向显存写入 `DEOPT_REQUEST` 并自觉挂起（不破坏其余请求）。主机发现异常后微冻结管线，唤回权重并单独重放。这用万分之一的局部挂起代价换取平稳期的全域零开销。

### 9.5 尾段就地观测 (Zero-Copy Paged Telemetry)

- **绝对零数据搬移**: **严禁** 在热路径外由 CPU 启动独立的轮询队列或配置无锁环形队列 (RingBuffer) 同步观测指标。
- **后置收集 (Epilogue Instrumentation)**: 必须在 RmsNorm / Softmax 网尾层级的底层 PTX/ASM 尾段，捎带计算出该 Batch 的 `系统熵 (Entropy)`、`概率质心 (Centroid)`、`跨层能量差 (Residual Delta)`。
- **页头常驻 (In-Place Memory Paging)**: 所有的探视信息伴随原流水线写入 KV Cache 时，直接利用普通内存存储指令（`STG`）刻入 Token 闲置的物理块头 (Header Padding)，让宿主机使用无感后台低频轮询机制进行拾取。

### 9.6 L1i 指令缓存预算协议与代码段分割 (ARCH-L1I-BUDGET)

> **核心约束**: §9-§17 定义了大量优化机制（MoE 分发、Compact/Scatter、残差注入、Guardrail 探针、TurboQuant FWHT、推测解码验证等）。如果这些机制的代码全部平铺到一个 Mega-Kernel 中，**条件分支的 `cmp/jmp` 链 + 冷路径代码体会撑爆 L1i**，导致每步 decode 发生指令缓存颠簸 (I-Cache Thrashing)，性能下降 30-50%。
>
> **铁律**: 运行期热路径的指令足迹 (Instruction Footprint) 必须 **≤ 80% L1i 容量**。剩余 20% 留给硬件预取和 alignment gap。

#### 9.6.1 L1i 物理预算

| 平台 | L1i 大小 | 可用预算 (80%) | 说明 |
|------|---------|---------------|------|
| x86_64 (Intel/AMD) | 32 KB | 25.6 KB | 主流服务器/桌面 |
| Apple M-series | 64-128 KB | 51.2-102.4 KB | 统一内存架构，L1i 较大 |
| ARM Neoverse V2 | 64 KB | 51.2 KB | 服务器 ARM |
| GPU SM (指令缓存) | ~4-8 KB/SM | 3.2-6.4 KB | GPU L0 instruction cache，极度紧张 |

#### 9.6.2 三段式代码段分割 (Code Section Tiering)

**核心原则: 变体特化 (Variant Specialization) 替代运行时分支 (Runtime Branching)**

不编译一个"万能 Mega-Kernel"（内含 `if moe_active`、`if guardrail_active` 等条件链），而是编译 **N 个场景特化变体**，每个变体仅包含该场景需要的代码路径。

```
┌─────────────────────────────────────────────────────────────────────┐
│ .text.hot — L1i 常驻 (≤80% L1i, 必须)                              │
│                                                                      │
│ 每个 Variant 仅包含其场景的必要指令：                                │
│                                                                      │
│ Variant-Dense:                                                       │
│   RmsNorm → QKV GEMM → RoPE → Attention → Residual Add →            │
│   Gate GEMM → SiLU → Up GEMM → Mul → Down GEMM → Residual Add       │
│   + Compact/Scatter index buffer (~512B code)                       │
│   + Telemetry epilogue: 3-5 条 STG per layer                        │
│   Total: ~18-22 KB ✅                                                │
│                                                                      │
│ Variant-MoE:                                                         │
│   [同上 Attention 侧] → MoE Gate → TopK → Expert FFN×K →            │
│   WeightedSum → Residual Add                                         │
│   + Perfect Hash Jump Table (§12.7.3, 立即数内嵌)                   │
│   + Expert prefetch hint (1 条 PREFETCH per expert)                  │
│   Total: ~20-24 KB (专家权重不进 SMEM, 仅 L2 预取) ✅               │
│                                                                      │
│ Variant-SpecVerify:                                                  │
│   [同 Dense/MoE] + Tree Attention Mask (CSR indptr 立即数偏移)       │
│   + KV Commit/Rollback trampoline (~256B)                            │
│   Total: ~22-26 KB ✅                                                │
├─────────────────────────────────────────────────────────────────────┤
│ .text.warm — L2 常驻, 通过 NOP Trampoline 按需拉入 L1i              │
│                                                                      │
│ - Compact/Scatter 完整逻辑 (vcompress/warp prefix sum)               │
│ - TurboQuant FWHT 旋转序列 (~2 KB per insertion point)              │
│ - MoE 异步预取完成回调 (~512B)                                      │
│ - Chunked Prefill 分轨 Attention (prefill SMEM 布局, ~4 KB)         │
│                                                                      │
│ 触发机制: .text.hot 中的 NOP Trampoline 被 §9.2 原子覆写为           │
│           `jmp .text.warm+offset` — 单次跳转，零条件判断             │
├─────────────────────────────────────────────────────────────────────┤
│ .text.cold — L3/DRAM, 长跳转 (Long JMP)，几乎不执行                 │
│                                                                      │
│ - Deopt Handler (Uncommon Trap 恢复, §15.4)                         │
│ - Guardrail Safety Veto 处理 (§16.4)                                 │
│ - Shadow KV Projector 推理 (§17.5)                                  │
│ - Early-Exit micro lm_head 推理 (§16.2)                             │
│ - Late-Fusion RAG 知识注入 (§16.1)                                  │
│ - SAGUARO NCCL all-gather 同步路径 (§17.10)                         │
│                                                                      │
│ 触发机制: thread block 写 DEOPT_REQUEST / SafetyHalt 标记,           │
│           挂起自身。主机端微冻结后执行 cold path, 单独重放。           │
│           热路径零开销 — 连跳转指令都没有。                          │
└─────────────────────────────────────────────────────────────────────┘
```

#### 9.6.3 变体选择决策 (Variant Selection at Dispatch-Time)

**关键: 变体选择发生在批构建阶段 (Dispatch-Time)，不在 Mega-Kernel 执行时。**

```
build_batch() {
    1. 收集 batch 中所有请求的属性:
       - has_moe_layers?  → 需要专家分发代码
       - has_guardrail?   → 需要 guard 探针代码
       - is_spec_verify?  → 需要 tree attention 代码
       - has_rag_injection? → 需要残差注入代码
       - chunk_size?      → 映射到 Golden Size

    2. 查找 VariantRegistry:
       key = (model_arch, moe_enabled, guardrail_enabled,
              spec_mode, rag_enabled, golden_size)
       → 命中: 返回已编译的 CompiledVariant
       → 未命中: 同步编译新变体 (缓存到 ModelJitCache)

    3. launch Mega-Kernel with selected variant
}
```

**L1i 预算保证**: 每个 CompiledVariant 在编译时计算 instruction footprint。如果超过 80% L1i 预算，编译器自动执行:
- 拆分: 将 MoE Expert FFN 降级为 .text.warm (NOP Trampoline 唤起)
- 裁剪: 禁用低优先级 telemetry fusion points (§13.10-13.11)
- 拒绝: 返回 `Err(L1iBudgetExceeded)` — 要求减小 batch 或禁用部分机制

#### 9.6.4 NOP Trampoline 机制 (零开销冷路径入口)

热路径代码在可能触发冷路径的位置预留 **5-byte NOP Slide** (x86: `0F 1F 44 00 00`) 或 **2-byte NOP** (GPU: `NOP.S 0x0`)。

```
热路径 (编译时):
  RmsNorm → GEMM → ... → Residual Add → [NOP SLIDE] → next layer

激活后 (§9.2 Hot JMP 覆写):
  RmsNorm → GEMM → ... → Residual Add → JMP .text.warm+offset → cold handler
  下一层继续从 .text.hot 执行 (长 JMP 返回)
```

**L1i 影响**: NOP Slide 是 5 字节，在 L1i 中与前后指令连续缓存，不引发 miss。只有当 NOP 被覆写为 JMP 且目标在 .text.warm 时，才会发生一次 L1i→L2 的指令取回。

#### 9.6.5 GPU 指令缓存特殊约束

GPU SM 的 L0 instruction cache 仅 4-8 KB。GPU 端的代码段分割策略:

1. **Tile 级代码复用**: 每个 Thread Block 的内循环 (GEMM K-loop) 必须 ≤ 2 KB，确保循环体 100% 驻留 L0
2. **跨 Tile 长跳转禁止**: GEMM 微内核的 K-loop 内禁止任何 `BRA` 到 .text.warm — 所有分支目标必须在同一 L0 缓存行
3. **冷路径 = 主机端执行**: GPU 的 Deopt/Guardrail/Safety Veto 全部标记 → 写 flag → 挂起 → 主机端处理。GPU 端不包含冷路径代码体

#### 9.6.6 与 §12.4 黄金装筒的融合

黄金装筒 (Golden Sizes) 不仅约束数据形状，也约束 **代码形状**:

| Golden Size | 编译的 Variant 数 | 每 Variant 指令足迹 | L1i 状态 |
|-------------|------------------|-------------------|---------|
| SEQ=1 (decode) | 4 (dense/moe/spec-verify/spec-draft) | 18-26 KB | ✅ 全部常驻 |
| SEQ=64 (chunk) | 2 (dense/moe) | 22-28 KB | ✅ 常驻 |
| SEQ=128+ (prefill) | 2 (dense/moe) | 24-30 KB | ⚠️ 需检查预算 |

**装筒 = 代码装筒 + 数据装筒**: §12.4 的 Golden Size 既是张量维度的塌缩目标，也是指令足迹的预算上限。每个 Golden Size 的所有 Variant 预编译产物总和 ≤ L2 cache (256 KB-2 MB)，确保 Variant 切换时 L1i miss 仅需一次 L2 取回。

---

## §10 Chunked Prefill 无限上下文支撑架构 (ARCH-CHUNKED-PREFILL)

> **实现状态**: ✅ IMPLEMENTED — chunked_prefill.rs(482L/10 tests), compact.rs(286L/5 tests), batcher.rs 集成
> **核心使命**: 让系统能够处理几乎无限大的上下文（10M+ Context），同时保持 Decode 请求的零等待延迟。

### 10.1 基于请求形态的交织调度 (Interleaved SEQ-Aware Scheduling)

传统 Continuous Batching 将 Prefill（`SEQ > 1` 的长序列填充）与 Decode（`SEQ = 1` 的逐字生成）进行阶段隔离，导致：
- GPU 利用率方差达 35%（Memory Bound vs Compute Bound 交替）
- Decode 请求必须等待长文本 Prefill 完成（Tail Latency 恶化至 ~200ms P99）

**Gllm 的破局法则**：将无限大的 Prefill 长文强制切成固定大小的物理切片（Chunk），与 Decode Token 交织塞进同一个 Batch，**解码请求永远零等待**。

- **精度的物理隔离壁 (Attention Phase Isolation)**：Prefill Chunk 的 Softmax 分布相对平坦，而 Decode Token 的 Softmax 高度尖锐。在极低精度下，这两种注意力分布模式在同一 Shared Memory 中计算会发生严重的精度交叉污染。因此，**在 Attention 阶段，Prefill Chunk 和 Decode Token 必须被物理分轨调度到不同的 Thread Block 组和 SMEM 分区执行**（复用 §12.7 的 Fat-Binary 跳表实现零开销切换）。FFN 阶段因无分布敏感性允许合流。

```
Chunked 调度（交织）:
┌─────────────────────────────────────────────────────────────┐
│ Batch 1:                                                    │
│   [Prefill-64, Decode-1, Decode-1, Decode-1, Decode-1,      │
│    Prefill-64, Decode-1, Decode-1, Decode-1, Decode-1]      │
│    ↑ Chunk 1       ↑ Decode 插槽 (优先)    ↑ Chunk 2        │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 切分策略

| 策略 | Chunk Size | 适用场景 |
|------|------------|----------|
| **固定切分** | 64/128 tokens | 通用场景 |
| **自适应切分** | 根据内存压力动态调整 | 显存受限 |
| **优先级切分** | 高优先级请求更大 Chunk | SLA 保障 |

### 10.3 调度流程

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 收集 Decode 请求 | 填充 decode_slots 直到达到配置上限 |
| 2 | 计算剩余预算 | `remaining_budget = max_tokens - decode_tokens` |
| 3 | 填入 Prefill Chunk | 从 prefill_queue 取出请求，按 chunk_size 切分 |
| 4 | 更新请求进度 | 完成的 Chunk 从 pending 减少 |
| 5 | 请求状态转换 | Prefill 完成后转为 Decode 请求 |

### 10.4 SplitFuse 废弃声明

> **⛔ REQ-SCHED-007**: SplitFuse 混批路径已被永久移除。不再支持拆分 QKV 与 Attention 融合的畸形计算流。`enable_splitfuse` 配置字段已弃用并锁定为 `false`。

### 10.5 验收指标

| 指标 | 目标值 | 测量方法 |
|------|--------|----------|
| Tail Latency (P99) | < 50ms | 混合负载测试 |
| GPU 利用率方差 | < 15% | 性能监控 |
| 吞吐提升 | +30-50% | 对比纯 Decode Batch |
| 内存开销 | < 5% 额外 | 显存监控 |

### 10.6 Batch Composition 算法 (ARCH-BATCH-COMPOSITION)

> **关联 REQ**: REQ-SCHED-003 (Continuous Batching), REQ-SCHED-016 (ChunkedConfig), REQ-SCHED-017 (BatchOrderPolicy)
> **关联架构**: §9.1 (Compact→Execute→Scatter), §12 (空间异构流片)
> **核心职责**: 定义调度层如何将 decode tokens 和 prefill chunks 组合成一个物理 batch，以及 compact 的触发与开销权衡。

#### 10.6.1 Token Budget 分配

每个 step 的 batch 容量由 `max_batch_tokens`（硬件上限）和运行时内存压力共同决定。

```
total_budget = max_batch_tokens × memory_pressure_ratio
decode_budget = min(
    decode_ready_count,                          // 实际 ready decode tokens
    floor(total_budget × decode_ratio_cap)       // decode 占比上限 (默认 0.6)
)
prefill_budget = total_budget - decode_budget
```

**关键约束**:
- `decode_ratio_cap = 0.6`：decode 最多占 60% 预算，保证 prefill 持续进展（避免 decode 饥饿 prefill）
- decode tokens 严格按 `BatchOrderPolicy` (默认 `StrictRequestIdOrder`) 排序后填充
- Chunk 大小由 `AdaptiveChunkPolicy` 根据 `l1_available_ratio` + `concurrent_reqs` + `remaining_prefill_tokens` 动态决定

#### 10.6.2 Batch Composition 五步流程

```
Step 1: 收集 ready decode tokens
  - 遍历 active sequences，收集 needs_prefill() == false 的请求
  - 按 BatchOrderPolicy 排序（默认 RequestId 升序）
  - 每个 decode 请求贡献 1 token

Step 2: 填充 decode slots
  - 按排序顺序逐个填入 decode tokens
  - 直到 decode_budget 用尽或所有 ready decode 已填入

Step 3: 计算剩余 prefill_budget
  - prefill_budget = total_budget - actual_decode_used
  - 如果 prefill_budget == 0：跳过 Step 4，输出纯 decode batch

Step 4: 填入 prefill chunks
  - 从 prefill_queue 按优先级取请求
  - 每个请求按 adaptive_chunk_size 切分
  - 填入直到 prefill_budget 用尽或 max_chunks_per_batch 达到上限
  - 完成全部 prefill 的请求转入 decode 状态

Step 5: 生成 BatchManifest
  - 记录每个 slot 的 request_id, type(Decode/PrefillChunk), token_range
  - 计算 compact_required 标志（见 §10.6.3）
  - BatchManifest 传递给 Executor 执行
```

#### 10.6.3 Compact 决策模型

§9.1 和 §17.6 定义了 Compact→Execute→Scatter 的硬件实现。本节定义**何时触发** compact。

**测量点**: BatchManifest 生成后，计算 batch 内的 lane 利用率。

**触发条件** (必须同时满足):
1. `waste_ratio = (batch_size - active_count) / batch_size > 0.25`
2. `active_count >= min_compact_threshold` (默认 4，避免极小 batch 的 compact 开销占比过高)
3. compact 发生在 GEMM op 级别，**禁止在 Attention op 上做 compact**（attention 是 memory-bound，compact 无法节省 memory bandwidth，反而增加数据搬移）

**开销模型**:
```
compact_cost = 2 × active_count × elem_size × cache_line_latency
             // compact 一次 + scatter 一次 = 两次内存操作
saved_flops  = waste_ratio × total_flops
decision     = compact_cost < saved_flops × flops_to_mem_ratio
             // flops_to_mem_ratio = peak_flops / peak_bandwidth (hardware-specific)
```

**与 PagedAttention 交互**:
- Compact 操作在 hidden_states 层面（GEMM 输入），不影响 KV cache page 管理
- Scatter 写回 hidden_states 原始 batch 位置（非 KV cache）
- KV cache 写入发生在 scatter 之后，由各 request 独立的 KV commit 完成
- 跨 page 边界：scatter 本身不涉及 page 边界（操作 hidden_states），KV write 由 PagedAttention 正常处理

#### 10.6.4 Sub-Batch 分发算法

当 batch 内请求形态分歧较大时（§12.1），将 batch 切分为多个 sub-batch 并行执行。

**分类维度**: 三维分类 key
```
SubBatchKey = (seq_len_range, exit_state, moe_active)
```
- `seq_len_range`: 按 §12.4 黄金装筒规则映射到最近的 golden size
- `exit_state`: Normal / EarlyExit / SpeculativeVerify
- `moe_active`: true / false（MoE 门控是否激活）

**分发规则**:
1. 按 SubBatchKey 分组，每组为一个 sub-batch
2. `min_sub_batch_size = 4`：低于此值的 group 回退到 §12.2 Ragged Compaction（不独立发射）
3. GPU: 按 sub-batch 比例分配 SM 数量（gridDim 按 sub-batch 大小缩放）
4. CPU: 按 NUMA node 绑核分配（AMX 核心处理密集计算，AVX 核心处理低精度）

**与 HGAL 交互**:
- Sub-batch 分发发生在 BatchManifest 生成之后、Executor 执行之前
- HGAL 优先级决定请求在 batch 内的排序，不影响 sub-batch 分类
- Eviction 决策与 sub-batch 分发解耦（eviction 在 batch 级别，sub-batch 在执行级别）

---

## §11 TurboQuant 2.0 运行时数学精度优化体系 (ARCH-TURBOQUANT)

> **实现状态**: ✅ IMPLEMENTED — quant.rs(1231L/24 tests), dual_track.rs(605L/17 tests), fp8.rs(203L/5 tests), turboquant.rs(345L)
> **学术依据**: SpinQuant (ICLR 2025), KurTail (2025), QuIP# (ICML 2024), RaBitQ (SIGMOD 2024), KIVI
> **核心哲学**: "无损"不是"权重逼近原值"，而是"推理过程中内积/输出的期望与全精度一致"。TurboQuant 是 gllm 在前向传播中执行的一组运行时数学优化，通过在线旋转、非对称 KV 量化、无偏修正等手段，使推理精度在任意量化权重格式上逼近数学无损。
> **定位**: gllm 加载 SafeTensors/GGUF/ONNX 全格式权重，TurboQuant 优化推理过程本身——不管权重来自哪里，不管权重用什么工具量化的。

### 11.1 在线旋转插入点 (Online FWHT Insertion Points)

前向传播中存在 3 个非线性边界（Softmax、SwiGLU 门控乘法、RoPE），旋转变换无法穿越这些非线性层，必须在运行时执行 Fast Walsh-Hadamard Transform (FWHT)。与 Mega-Kernel 铁律完全兼容: 不增加新的 Kernel Launch（内联在 Mega-Kernel 指令流中），不引入 if-else 分支（FWHT 是固定指令序列），复杂度 $O(d \log d)$ 远低于 GEMM 的 $O(d^2)$。

| 位置 | Mega-Kernel 阶段 | 白嫖路径 |
|------|-----------------|---------|
| Softmax(QK^T) V 输出之后 | Attention Epilogue 尾段内联 | 数据在寄存器/SMEM，无额外全局内存读写。$R^T$ 已离线吸收进 $W_o$ |
| SwiGLU(Gate) Up 输出之后 | FFN 中间态 Epilogue 内联 | 数据在寄存器。$R^T$ 已离线吸收进 $W_{down}$ |
| RoPE(K) 存入 KV Cache 之前 | KV Write 阶段内联 | 与 Epilogue Zero-Copy Paged Header Write 共享同一条 STG 指令流 |

### 11.2 KV Cache 非对称量化 (Asymmetric KV Quantization)

基于 KIVI/KVQuant 研究: Key 和 Value 的离群点分布特性截然不同。

| 维度 | 离群点特征 | 量化粒度 | Scale 来源 |
|------|-----------|---------|-----------|
| **Key Cache** | 集中在特定通道（跨 Token 稳定） | Per-Channel | 离线校准常数（零运行时开销）|
| **Value Cache** | 集中在特定 Token（跨通道稳定） | Per-Token | KV 写入时寄存器内 reduce_max |

**Attention Sink 保护**: 前 $N$ 个 Token（默认 $N=4$）保留 FP16 全精度。Sink 判定从 Epilogue Telemetry 的 Entropy/Centroid 数据推导，无需新增探测代码。

### 11.3 无偏性保证 (Unbiased Inner Product Estimation)

受 RaBitQ (SIGMOD 2024) 启发，在 Attention $QK^T$ 计算中引入修正因子: $\widehat{QK^T} = QK^T_{quant} \cdot C_1 + C_0$。

- $\|v\|$（修正因子输入）: 从 RMSNorm 免费白嫖，$\|v\| = \text{RMS} \cdot \sqrt{d}$
- 量化前后内积: 在量化循环中追加 1 条 FMA 指令
- 理论误差界: $O(1/\sqrt{D})$。$D=4096$ 时约 $1.5\%$，$D=8192$ 时约 $1.1\%$

### 11.4 白嫖全景与运行时净开销 (Freeloading Summary)

gllm 在推理过程中执行的全部 TurboQuant 运算及其净开销：

| 运算 | 白嫖来源 | 净开销 |
|---------|---------|--------|
| Group Scale 计算 | RMSNorm 规约尾段追加 max | **2-4 条指令/层** |
| Attention 后 FWHT | Attention Epilogue 内联 | **$O(d\log d)$ 算术** |
| SwiGLU 后 FWHT | FFN Epilogue 内联 | **$O(d\log d)$ 算术** |
| KV Cache 写入 K 旋转 | KV Write 阶段内联 | **$O(d\log d)$ 算术** |
| KV Cache V per-token scale | KV 写入时寄存器内 reduce_max | **极低** |
| RaBitQ 修正因子 | RMSNorm 范数 + FMA | **1 条 FMA** |
| Sink Token 检测 | Epilogue Entropy/Centroid | **零** |

**结论**: TurboQuant 的全部运行时开销 = 3 个在线 FWHT（每个 $O(d\log d)$）+ 每层若干条 max/FMA 指令。其余全部被 Mega-Kernel 的 Epilogue 白嫖基础设施吃掉。

### 11.5 双轨显存池 (The Dual-Track Memory Pool)

重构 `KvCacheConfig`，全面淘汰 `dtype_size`。由 `GlobalMemoryManager` 申请物理隔离的两轨架构:

| 轨道 | 物理精度 | 职能 |
|------|------|------|
| **主池** | 极低精度 (3-4bit) | 组级缩放由 Epilogue 快递，KV 按 11.2 非对称量化 |
| **校验池 (QJL)** | 1-bit | XNOR 残差掩码阵列 |

**多卡同步红利**: PCIe Swap 和跨卡 RDMA 同步 KV 时，仅需传输原 FP16 内存量纲的 25%（4x 压缩），突破总线墙。

---

## §12 空间异构流派与动态块式计算图 (ARCH-SPATIAL-DISAGGREGATION)

> **实现状态**: ✅ IMPLEMENTED — profiler.rs(600L/6 tests), histogram.rs(395L/8 tests), ragged.rs(693L/14 tests), sub_batch.rs(507L), variant_registry.rs(481L)
> **关联**: 会话 6e743114 §14
> **核心使命**: 解决大吞吐量下，个体动态分歧与 Batch 同步执行的木桶效应。

### 12.1 核心维度的空间异构流片 (Sub-Batching by Graph Shape)

不同的 CPU/GPU 核心（Cores / SMs）物理分区，同时运行着**不同拓扑结构的 JIT 图**：

1. **零散归类 (Sub-Batching)**: 调度器发现本批 128 个请求中：60 个可跳过注意力（Shape A），30 个死神经元走 INT8 窄网（Shape B），38 个需完整 FP16 密集网络（Shape C）
2. **多流分发 (Multi-Stream Spatial Dispatch)**: 不再取最坏形状统一掩码，而是切割为三个子批次，并行发射到不同硬件分片
3. **硬件资源软分区 (Software-Defined Core Partitions)**:
   - **GPU 端**: JIT 代码生成时锁死 `gridDim` 与 `Block Size`，强制 Shape A 的 Kernel 只占用 SM[0-15]，Shape B 占用 SM[16-40]，Shape C 占用 SM[41-80]。共享 L2 和显存带宽，互不卡脖
   - **CPU 端**: 利用 `NUMA Node / Core Affinity` 绑核。AMX 核心处理密集计算，AVX 核心处理低精度稀疏映射

### 12.2 兜底回收 (Residual Compaction & Predication)

当零碎请求不足以构成独立 Sub-Batch（如某形状仅 2 个单孤 Token），回退到：
- **动态张量紧凑重排 (Ragged Compaction)**: `vcompress` / Warp Prefix Sum
- **硬件谓词掩码 (Predicate K-mask)**: 保持完整控制流用掩码位强行断电

**"小差异合流兜底、大分层异构分治、长尾期修补坍缩"** —— 这种三位一体的体系彻底补齐了 Gllm 面对百万级动态上下文（10M Context）+ 数百路并发时，算力无法充分释放的理论死角。

### 12.4 硬件感知型黄金装筒规则 (Hardware-Aware Shape Bucketing)

这是动态图形状隔离的“第一起因（First Cause）”，定义了系统如何根据不同的 **SEQ（上下文长度）** 或者 **异构请求状态** 生成**最佳集中性能状态图**：

- **底层时滞与拓扑探测 (Hardware Telemetry Probing)**：
    在 Load-Time（模型加载阶段）与 Autotuning 期间，**严禁**预设死板的数组（如 `[128, 512, 1024, 2048]`）作为 JIT 的静态 Bucket。
    系统必须依靠真实的“底层时滞探测器（Latency Probes）”测定在当前微架构（如 Ada Lovelace 或 Zen4）上：
  - 寄存器堆何时开始溢出 (Spilling out points)
  - 共享内存 (SMEM) 的满载率与占用悬崖 (Occupancy Cliffs)
  - L2 缓存 Thrashing 阈值

- **物理资源的离散塌缩 (State Graph Concentrating)**：
    由探测器探出的物理拐点圈定出仅有的几档“黄金尺寸”（Golden Sizes，例如针对某模型/显卡，探测出的最佳分桶刚好是 112、463 和 1011）。对于推理期任意连续离散的 SEQ 长度，系统不再在运行期进行软路由判断或全尺寸 JIT 编译，而是将这些差异化请求**全部强行映射/塌缩**到这几张 **只针对该硬件表现为“满血最佳集中性能”的静态 JIT 状态图** 上。

- **零退化原则 (Zero-Padding Degradation)**：
    当实际请求长度与“黄金尺寸”产生部分空隙时，**坚决禁止**使用低效的 Padding 补零机制（如为了对齐 128 强行补零，废算无效 FLOPs）。
    必须依靠 §12.2 定义的张量物理挤压（Ragged Compaction，如 `vcompress` 或 Warp Prefix Sum）以及硬件控制流谓词（Predicate K-mask），实现完全 0 周期浪费的“真填满”，确保生成的 Kernel 永远呈现 100% 寄存器命中率与管线吞吐率。

- **运行时装筒热演化 (Runtime Bucket Evolution)**：
    黄金装筒绝非 Load-Time 的静态死水。JIT Director Daemon 必须持续观测流量的 SEQ 分布直方图。如果负载发生时段性偏移，大量请求密集落在现有 Bucket 的缝隙中（导致挤压也无法挽回物理层面的微观浪费），JIT Director 允许在后台沙盒中**即时编译新的中间态 Bucket 变体（如 Bucket-96）**。编译完成后利用 §9.2 的原子覆写机制热插入跳表，并在 L1i 缓存重排中淘汰命中率 < 0.1% 的僵尸 Bucket。实现运行时无停机演化。

### 12.5 JIT 硅晶指令深层映射原则 (Silicon-Level Instruction Mapping)

全世代硬件能力（REQ-HARDWARE-SENSORS）不仅仅是调度约束，更必须下沉入 JIT Mega-Kernel 的汇编强映射：

- **x86/ARM 脉络 (Vectors & Converged Cores)**
  - **AVX10 P/E 收敛映射**: 面对 AVX10.1/10.2 混合大小核架构，JIT 必须探明 `256-bit Converged Vector` 屏障。在调度 E-Core 时，物理寄存器切片图自动塌缩至 256 宽，同时利用 `vp2intersect` 指令无缝下发稀疏张量的交叉表。
  - **AVX512 极限物理挤压**: 上述文的张量挤压（Ragged Compaction）在 x86 必须翻译为 `vcompress` 系列指令；利用全量的 31 个通用寄存器 (APX) 彻底消灭内循环的 Spilling 行为。

- **NVIDIA GPU 脉络 (Hopper/Blackwell Evolution)**
  - **Hopper (SM90) 内存墙突围**: 绝对禁止使用传统 `LDG` 执行 KV Cache 离散读取！JIT 代码必须生成 `TMA (Tensor Memory Accelerator)` 配置包，并结合 `WGMMA` 与 `cuda::barrier` 实现生产者-消费者（Thread Block Cluster）内存直接多播传输（L2 mcast）。
  - **Blackwell (SM100) 精度原生力**: 针对 W4A4 / W4A8 TurboQuant 适配，编译器直写针对 `FP4 / FP6 Native Tension` 优化的 MMA 汇编指令，抛弃任何高精度的模拟或溢出防范开销，发挥 Block Scale 底层机制的最大吞吐。

- **AMD GPU 脉络 (CDNA)**
  - 利用 CDNA3 管线的 XCD/GCD 拓扑屏障，确立本地物理隔离。配合内置的 `WMMA` 指令进行张量吞吐提速。

### 12.6 硬件探测→IR 强约束变量体系 (MicroArch-to-IR Constraints)

> **关联**: 本文档 §12 硬件探测, resolved.10 §2.2
> **核心法则**: 废弃散乱的指令集条件判断。所有加载期的硬件探测结果必须严格坍缩为对底层 JIT 编译器的 **强数学约束变量组（Compiler Constraints）**，确保 JIT 逻辑与物理实体芯片解耦。

#### 物理传感器指标

| 传感器类目 | 探测项 | 说明 |
|-----------|--------|------|
| **缓存与拓扑** | `L1i` (指令缓存大小) | 防止大 Bucket (如 `seq=1024`) 的指令平铺导致 L1i 缓存颠簸。触达 `l1i_size * 0.8` 时 JIT 退化为 `jmp` |
| | `L2_Cache` | GPU 分级驻留锚点 |
| | `CCX/NUMA` 边界 | AMD CCD 的 L3 分区嗅探，规避跨片通信 |
| | `TLB Entries` | 防止大显存映射的 TLB Thrashing |
| **x86 极化阵列** | `AMX`, `APX` (31 GPR), `VNNI/AVX_VNNI_INT8` | 见 §12.5 |
| **ARM 极化阵列** | `SVE2`, `SME/SME2` (`ZA_Array_Size`) | 及 Streaming SVE 模式阈值 |
| **总线协议** | GPU `TMA` (Hopper 异步内存张量加速) | 见 §12.5 |
| **RDMA/Scale-Out** | `nic_bandwidth_gbs`, `rdma_latency_us` | 跨机 NUMA 拓扑必须包含网卡带宽和 RDMA 延迟 |

#### IR 约束变量输出 (Target Execution Topology)

传感器数据转化为 CompilerGraph 直接可消费的环境变量：

| 约束变量 | 说明 | 典型值 |
|----------|------|--------|
| `max_gpr_count` | 控制寄存器溢写（Spill）阈值 | 普通 CPU=15, APX=31 |
| `optimal_tile_bits` | 决定 JIT 平铺展开的二维尺寸 | 启用 AMX/SME 阵列时极速扩大 |
| `native_int4_dot` | 标定是否可通过 VNNI/SVE2 直接下发硬件解包 | true/false |

#### RDMA Pipelining 融合约束

当引擎跨越单机时，Chunk 的切分块大小必须满足：

$$T_{\text{compute}}(\text{chunk}) \geqslant T_{\text{rdma\_transfer}}(\text{chunk})$$

即单卡对该 Chunk 的计算时间 ≥ 跨节点 RDMA 参数投递时间，从而实现 Pipelining 掩盖网络延迟。此约束由 `rdma_latency_us` 和 `nic_bandwidth_gbs` 共同确定。

### 12.7 终极静态化极境 (The Ultimate Staticization Limits)

在现有 JIT Mega-Kernel 和 TurboQuant 的架构底盘之上，为了彻底榨干硅晶的最后一滴血，Gllm 将残留于内循环（Inner Loops）中极少量的“动态计算开销”进行终极抹杀。我们将把**运行期的最后一点弹性和分支判断，全部提权并折叠到内核启动前（Load-Time & Dispatch-Time）！** 

我们称之为 **“运行期状态 0 化，编译期变体无限爆发 (State Minimization vs. Variant Explosion)”** 架构原则。

#### 1. 立即数硬编码步长 (Constexpr Memory Strides)
禁止 JIT 核内存在计算连续物理步长的动态计算（如 `seq_len * head_dim`）。
基于 §12.4 黄金装筒（Golden Sizes），因形状已是编译期定理，编译器直接将被截断好的张量偏移量化作**立即数常量 (Immediate values)**，原生内嵌至 PTX/ASM 访存底码中。内循环中彻底消杀针对内存地址重算的整数数学（`IMUL` 等）。

#### 2. 无分支极限循环展开 (Zero-Branch Full Unrolling)
依据探测约束变量 `optimal_tile_bits` 以及硬件宽度（如 AMX 的 1024-bit 算盘），编译器暴力解开并**全量抹平循环体 (Fully Unrolled FMA)**。
生成的该类型装筒变体中，不再存留任何属于控制流检查的对比器（`cmp`）和跳转界限分支（`br/jmp`），达成流水线（Pipeline）指令乱序执行（OOO）的无阻塞轰炸极限。

#### 3. 完美哈希跳表分轨 (Perfect Hashed Jump Tables)
针对诸如 MoE 路由分发或多意图抽离（Semantic Anchors），废除内循环中所有的“字典映射查询”。
编译器通过为分发任务生成 $O(1)$ 的无冲突完美哈希表，将其异变为**汇编级别基于步进指令的表层跳转（Static Switch ASM）**。在 `Mega-Kernel` 启动进入 Thread Block 的第一纳秒，瞬间物理踹走进入属于该任务/专家的独立纯净汇编层。

#### 4. 跨机 RDMA 的显存绝对硬绑定 (Static Memory Topologies)
在极化并行（Pipeline/Tensor Parallelism）部署体系中，特定的网卡通信队列对（QP）与特定 GPU (SM) 物理锁死绑定（Pinning）。
不需要再查地址树，JIT 解析图时直接将向远端投射地址和网络目标编码成恒定的双轨显存池物理只读指针，将通信网络操作像本地总线定址一样直呼。

#### 5. 量化隔离与模型只读图折叠 (Lazy JIT Variant & Graph Folding)
- **按需生成的量化泛型**：摒弃传统静态算子中的 `if (dtype == INT4)` 判断。在 Load 模型时权重 DType 精度与计算路径已定。JIT 编译器只精确生成**与该层目前物理类型一致的单特化汇编代码**（例如只生成带 SM100 FP4 的微指令汇编），运行时不需要推断，物理底层结构便保障了精度的零开销匹配。
- **系统静态前缀折叠 (Static KV Prefill Folding)**：面对数十万 token 仍雷打不动的安全审核机制和 Prompt，由编译器（AOT 层面）运算后，化作只读状态流。不用再去查询 KV，此部分的树直接作为“固化的计算缓存”，彻底剥夺访问带宽需求。

---

## §13 Epilogue 白嫖网络 — 全链路物理融合 (ARCH-EPILOGUE-FUSIONS)

> **实现状态**: ✅ IMPLEMENTED — 12个白嫖融合点全部实现 (§13.10 Embedding ‖embed‖₂ 为最后一公里, 2026-04-05 完成)
> **关联**: 会话 6e743114 §4, 会话 全链路审计
> **核心原则**: 所有的特征检测必须"寄生"在上游核函数的数学尾段（Epilogue），严禁单独发起采集循环。
> **审计基准**: 从 Embedding → Layer Loop → lm_head 全管线，识别 11 个白嫖点（4 已规划 + 7 新发现）。

### 13.1 融合 1：Gate-First 掩码层跳过 (Masked CondGEMM)

取代"完整计算后剪裁"的做法：
- 在 JIT Mega-Graph 的 FFN 阶段：经过 Gate GEMM 后，Epilogue 立刻评估 $\text{SiLU}(g)$
- 探测到死神经元（如 $> 50\%$ 列失效）时，立即通过硬件掩码跳过后续的 Up GEMM 和 Down GEMM 通道
- **FLOPs 当场砍掉 40%**

### 13.2 融合 2：Softmax 质心引导预取 (Centroid-Guided Prefetch)

在 Chunked Prefill 阶段面临超大 Context，KV Cache 不可避免发生 L2 未命中：
- **白嫖触发器**: 利用第 N 层 Softmax 的 Epilogue 提取概率高地（质心 / Centroid）
- **JIT 联动**: 将质心坐标压入异步内存队列，触发 `cuMemPrefetchAsync`（或 CPU/AMX 的软件预取指令），让第 N+1 层将要关注的 KV Chunk 越过系统总线**提前驻留**

### 13.3 融合 3：残差旁路 (Residual Bypass)

- **白嫖触发器**: Residual Add 循环的 Epilogue 免费测算跨层能量方差 $\Delta \rho$
- **JIT 联动**: 如果 $\Delta \rho < 0.001$，直接触发底层的汇编跳转 `jmp END_OF_LAYER`
- 不管是 Chunk 还是单步 Decode，整个 Layer Q/K/V 和 FFN 的核函数被暴力切掉，**省下 ~15% 全局算力**

### 13.4 融合 4：KV Write FWHT 旋转 (Runtime FWHT Injection)

- **白嫖触发器**: KV Cache 写入阶段数据已在寄存器，追加 $O(d \log d)$ FWHT 在线旋转
- **JIT 联动**: 旋转后 KV 值的离群点被均匀分散，后续层的 Attention 计算数值稳定性提升
- **净开销**: $O(d \log d)$ 算术（远低于 GEMM 的 $O(d^2)$），内联在 KV Write 指令流中

---

### 全链路审计：7 个新发现白嫖点 (ARCH-EPILOGUE-EXTENDED)

> **审计时间**: 2026-03-29
> **审计方法**: 逐阶段分析推理管线中"数据在寄存器/共享内存停留但被直接丢弃"的位置

### 13.5 [P0] 融合 5：SiLU 死神经元掩码 (Dead Neuron Masking)

> **前置条件**: §13.1 Gate-First Skip 的信号来源
> **实现位置**: `gllm-kernels/src/compiler/codegen/x86_64.rs:12642-12680`（silu 实现）

在计算 $\text{SiLU}(x) = x \cdot \sigma(x)$ 时，sigmoid 值已在寄存器中：
- **白嫖触发器**: 追加 1 条 `vcmpps` + 1 条 `vphaddd`，统计 $\sigma(x) < \varepsilon$ 的元素数量
- **免费产物**: 死神经元计数（该列对 FFN 输出无贡献）
- **下游消费**: 如果死神经元 > 50%，触发 §13.1 Gate-First Skip，跳过 Up/Down GEMM
- **额外指令**: ~3 条 SIMD
- **关联**: §13.1 的前置信号源，替代已删除的 Amax 运行时检测（§14.1）

### 13.6 [P0] 融合 6：MoE Gate 命中计数 (Expert Hit Counter)

> **前置条件**: §15.4 冷板凳 Deopt 的信号源
> **实现位置**: `gllm-kernels/src/compiler/codegen/x86_64.rs:3820-3980`（TopK 实现）

Gate 概率经 softmax 后，TopK 线性扫描已在遍历所有概率值：
- **白嫖触发器**: 在 TopK 扫描循环中追加 1 条原子 `add`，将选中的 expert ID 写入共享计数器
- **免费产物**: 每个 expert 的命中率统计
- **下游消费**: JIT Director Daemon 轮询计数器，持续零命中的冷板凳专家触发 §15.4 Uncommon Trap 物理封杀
- **额外指令**: ~2 条（1 条 atomic add + 1 条条件判断）
- **优势**: 替代 SPEC 原设计中"JIT Director 扫描 KV Page Header"的方案，精度更高、延迟更低

### 13.7 [P1] 融合 7：GEMM 行级激活统计 (Row-wise Activation Stats)

> **前置条件**: §13.5 死神经元检测的扩展
> **实现位置**: `gllm-kernels/src/compiler/codegen/x86_64.rs:9301-9344`（`emit_epilogue_on_accumulators_inner`）

K-loop 结束后累加器寄存器（ymm/zmm）包含完整 GEMM 输出：
- **白嫖触发器**: 在 `emit_trace_on_accumulator` 写回前追加统计规约
- **免费产物**: 行级 $\|row\|_1$（1 条 `vphaddd` + 1 条 `vpaddd`）、行级 max（1 条 `vpmaxsd`）、行级 min（1 条 `vpminsd`）
- **下游消费**: 为 §13.5 Gate-First Skip 提供"是否死神经元"的前置判断；为后续层融合决策提供前驱张量分布特征
- **额外指令**: ~10 条 SIMD
- **关联**: 替代 §14.1 已删除的 Amax 检测，提供更丰富的分布信号

### 13.8 [P1] 融合 8：RmsNorm per-channel Scale (KIVI 量化信号)

> **前置条件**: §11 TurboQuant KV 非对称量化的 per-channel scale 来源
> **实现位置**: `gllm-kernels/src/compiler/codegen/x86_64.rs:10324-10423`（RmsNorm 规约阶段）

RmsNorm 的规约循环已在做 $\sum x^2$：
- **白嫖触发器**: 追加 1 条 `vmaxps` 记录逐通道绝对值最大值
- **免费产物**: per-channel scale = $\max(|x|)$，用于 KIVI 的 K per-channel 量化（§11.2）
- **下游消费**: 直接传入 KV Cache 写入阶段，替代独立的 scale 计算 kernel
- **额外指令**: 1 条 SIMD
- **关联**: 与 §11.2 KIVI 非对称量化联动，消除独立的 scale 计算开销

### 13.9 [P1] 融合 9：Softmax 锐度与 Attention Sink 检测 (Sharpness + Sink Detection)

> **前置条件**: §13.2 质心预取的扩展
> **实现位置**: `gllm-kernels/src/compiler/codegen/x86_64.rs:1880-1921`（Softmax max/sum 计算）

Softmax 的 max（`ymm4`）和 sum（`ymm5`）已在寄存器中：
- **白嫖触发器**: max 值本身免费可取；追加 1 条除法得到 max/sum 比值
- **免费产物**:
  - **max 值**: Attention Sink 检测（max 极高 → 当前 token 是 sink，保留全精度）
  - **max/sum 比值（锐度）**: 接近 1 → 尖锐关注某个 token；接近 $1/n$ → 均匀分散
  - **有效上下文长度**: `count(softmax > ε)`（追加 1 条 `vcmpps` + `vphaddd`）
- **下游消费**: Sink 检测 → §11.2 前 N 个 token FP16 保护；锐度 → 动态 chunk 大小调整
- **额外指令**: ~2 条
- **关联**: 扩展 §13.2 的质心预取，丰富 Softmax Epilogue 的产出

### 13.10 [P2] 融合 10：Embedding 范数初始化 (RaBitQ Bootstrap)

> **前置条件**: §11.3 RaBitQ 修正因子的初始值
> **实现位置**: `src/compat/decoder_forward.rs`（hidden_state 初始化）

Token embedding 查表后数据在 copy 路径上经过寄存器：
- **白嫖触发器**: 在 copy 循环中追加 reduce + sqrt
- **免费产物**: $\|embed\|_2$，作为第一层 RaBitQ 修正因子的初始值 $C_0$
- **下游消费**: 直接传入第一层 RmsNorm Epilogue，避免冷启动
- **额外指令**: ~5 条
- **关联**: 与 §11.3 RaBitQ 无偏修正联动

### 13.11 [P2] 融合 11：残差方向余弦 (Residual Cosine Similarity)

> **前置条件**: §13.3 残差旁路的精化指标
> **实现位置**: `gllm-kernels/src/compiler/codegen/x86_64.rs:5613-5651`（`emit_cross_layer_residual`）

Residual Add 的两个输入 $x_{in}$ 和 $x_{out}$ 都在寄存器中：
- **白嫖触发器**: 在现有 $\Delta\rho = \|x_{out}\| / \|x_{in}\|$ 基础上追加内积累加
- **免费产物**: 方向余弦 $\cos\theta = x_{in} \cdot x_{out} / (\|x_{in}\| \|x_{out}\|)$
- **下游消费**: 更精确的 Early Exit 指标——不仅仅看能量衰减，还看方向偏移。$\cos\theta > 0.99$ 且 $\Delta\rho < 0.01$ 时高置信度跳过
- **额外指令**: ~5 条（FMA + 除法）
- **关联**: 精化 §13.3 的旁路决策，减少误跳

---

### 全链路白嫖全景图

```
推理管线数据流（白嫖全景）

Token IDs
  │
  ▼
Embedding Lookup ──── §13.10: ‖embed‖₂ → RaBitQ 初始修正
  │
  ▼
═══ Layer Loop (×N) ═══
  │
  ├─ RmsNorm ─────── §11 ‖v‖ + §13.8 per-channel scale → KIVI 量化
  │
  ├─ Q/K/V GEMM ──── §13.7 行级 ‖row‖₁ + max → 死神经元检测
  │
  ├─ RoPE ────────── §13.4 FWHT 在线旋转
  │
  ├─ Softmax ─────── §13.2 Entropy + Centroid + §13.9 max + 锐度 → 预取 + Sink 检测
  │
  ├─ Residual Add ── §13.3 Δρ + §13.11 方向余弦 → Early Exit
  │
  ├─ RmsNorm2 ────── 同 RmsNorm
  │
  ├─ MoE Gate ────── §13.6 路由熵 + 命中计数 → Deopt 信号
  │
  ├─ FFN Gate SiLU ─ §13.5 死神经元掩码 → Gate-First Skip (40% FLOPs)
  │
  ├─ FFN Up × Down
  │
  └─ Residual Add ── 同上
  │
  ▼
═══ End Layer Loop ═══
  │
  ▼
lm_head GEMM ────── §13.7 logits 范数 → 采样策略调整
  │
  ▼
Output Token
```

---

### 13.12 硬件感知融合拓扑变换 (Hardware-Topology-Aware Fusion)

> **核心原则**: FusionRule 的 `plan()` 阶段根据 `DeviceProfile` 产生**不同的图拓扑**——不仅是同一算子的不同 codegen，而是节点数量、边连接、数据流方向根本不同。
> **关联**: §8 AttentionStrategy, §9 Mega-Kernel, §11 TurboQuant, §12 空间异构, gllm-kernels FusionEngine
> **硬件 SSOT**: REQ-HARDWARE-SENSORS 定义探测能力，本节定义探测结果如何驱动图拓扑

#### 13.12.0 硬件 Profile 分层定义

FusionRule 基于以下 Profile 层级产生拓扑决策：

**GPU 层级**:

| Profile | SM 范围 | 核心硬件能力 | 拓扑影响 |
|---------|---------|-------------|---------|
| **sm_100+ (Blackwell)** | sm_100+ | FP4/FP6 原生 Tensor Core; Block-scaled GEMM (per-block 缩放因子内置); TMEM 256KB/SM; tcgen05.mma; 2-CTA 协同 MMA; Thread Block Cluster | 权重全程 FP4 无反量化节点; Block-scaled 消除独立 Scale 节点; TMEM 替代 shared memory 做 attention tiling; 2-CTA 协同产生跨 CTA 边 |
| **sm_90 (Hopper)** | sm_90 | TMA 2D/5D prefetch; WGMMA 16×16×64; Warp Specialization (producer/consumer); FP8 native; cuda::barrier; L2 multicast | TMA 替代 cp.async; WGMMA 替代 mma.sync; Warp spec 产生双线程组子图 |
| **sm_80-89 (Ampere/Ada)** | sm_80-89 | mma.sync 16×8×16; cp.async 128B; BF16/TF32 Tensor Core; 32 寄存器/warp | mma.sync 主力; cp.async 异步预取; BF16 原生计算 |
| **sm_70-79 (Volta/Turing)** | sm_70-79 | wmma 16×16×16; 16 寄存器/warp; 无异步内存; FP16 Tensor Core | wmma 精度受限; 寄存器压力限制融合深度; 无异步预取 |

**CPU 层级 (x86)**:

| Profile | 核心硬件能力 | 拓扑影响 |
|---------|-------------|---------|
| **AVX10.2 + APX** | 256-bit 统一 SIMD; P/E 核混合感知; VP2INTERSECT; 31 GPR; BF16 256-bit 原生; VNNI-INT8 256-bit | 31 GPR 支持最深 epilogue 链 (≥8 ops 融合); P-core 走 256-bit SIMD 全量路径，E-core 走标量降级; VP2INTERSECT 硬件化 sparse mask 节点 |
| **AVX10.1 + APX** | 256-bit 统一 SIMD; 31 GPR; 基础 BF16/VNNI | 31 GPR 支持 6-8 ops epilogue; 缺 VP2INTERSECT，sparse mask 走软件 |
| **AVX-512 + AMX** | 512-bit SIMD (32 zmm); VNNI; VP2INTERSECT; BF16; AMX tile 8×8 BF16 | AMX tile GEMM 替代 BLIS 微内核; AVX-512 做 epilogue; 32 zmm 无溢出支持 8-op epilogue |
| **AVX-512 only** | 512-bit SIMD (32 zmm); VNNI; BF16; 无 AMX | BLIS 微内核做 GEMM; epilogue 用 AVX-512; 无 AMX tile 加速 |
| **AVX2** | 256-bit SIMD (16 ymm); FMA; F16C | 最保守拓扑: BLIS pack/unpack + 16 ymm epilogue 溢出到栈 |

**CPU 层级 (ARM)**:

| Profile | 核心硬件能力 | 拓扑影响 |
|---------|-------------|---------|
| **SME2 + SVE2** | ZA Array 2D 存储; streaming SVE 模式; outer product 矩阵乘; SVE2 可变长向量 | ZA array 消除显式 tile 管理节点; outer product 直接做 attention; streaming 模式切换开销影响跨层融合决策 |
| **SVE2 only** | 可变长 SIMD (128-2048 bit); SVE2 整数/浮点; 无 ZA array | 无 ZA 时回退 BLIS 风格 GEMM; epilogue 用 SVE2 predicated 向量 |
| **NEON** | 128-bit 固定 SIMD; ASIMD | 最保守: BLIS 微内核 + NEON epilogue; 类似 AVX2 体验 |

#### 13.12.1 Attention 子图拓扑

| 硬件 Profile | 子图拓扑 | 节点列表 | 关键差异 |
|---|---|---|---|
| **sm_100+** | FlashV4 Block-Scaled | `TMEMLoad_QKV → tcgen05.mma_QK(block_scaled) → Online_Softmax → tcgen05.mma_AV → Epilogue_FWHT → TMEMScatter_KV` | FP4 权重**全程无反量化**；block-scaled 内置 per-block scale 消除 Scale 节点；TMEMLoad 替代 TMA；2-CTA 协同拆分 Q 维度；TMEMScatter 一次写 KV |
| **sm_90** | FlashV3 Pipeline | `TMA_prefetch_QKV → WGMMA_QK → Online_Softmax → WGMMA_AV → Epilogue_FWHT → Scatter_KV` | TMA 2D prefetch; WGMMA warp 特化 (producer/consumer); FP8 辅助路径可选; Scatter 单 kernel 写 KV |
| **sm_80-89** | FlashV2 Tiled | `cp_async_pack_QKV → mma.sync_QK → Online_Softmax → mma.sync_AV → Epilogue_FWHT → DtoD_KV` | mma.sync 替代 WGMMA; cp.async 替代 TMA; DtoD 逐页写 KV |
| **sm_70-79** | wmma Tiled | `global_load_QKV → wmma_QK → Online_Softmax → wmma_AV → Epilogue_FWHT(可选) → DtoD_KV` | wmma 16×16×16 精度受限; 无异步预取; FWHT 收益小可跳过 |
| **SME2** | ZA Outer Product | `SVE2_RmsNorm → SME2_tile_load_QKV → RoPE(SVE2) → SME2_outer_product_QK → Softmax → SME2_outer_product_AV → Epilogue_FWHT → Direct_KV_Write` | ZA array 2D 存储消除 tile 管理节点; outer product 直接算 attention; streaming 模式切换在层边界 |
| **AVX10.2 + APX** | Tiled Naive + Deep Epilogue | `BLIS_QKV_GEMM(VNNI-256) → RoPE(AVX10) → Naive_Attn → Epilogue_FWHT(31GPR 深链) → Direct_KV_Write` | 31 GPR 允许 attention→FWHT→量化→写入 全融合; P-core 全速, E-core 降级为标量; VP2INTERSECT 硬件化 sparse mask |
| **AVX-512 + AMX** | Tiled + AMX GEMM | `AMX_QKV_GEMM → RoPE → AMX_Attn_GEMM → Epilogue_FWHT → Direct_KV_Write` | AMX tile 8×8 BF16 GEMM; 无 Flash tiling (CPU 无 SMEM); 32 zmm epilogue |
| **AVX-512** | Naive + BLIS | `BLIS_QKV_GEMM → RoPE → Naive_Attn(O(n²)) → Epilogue_FWHT → Direct_KV_Write` | BLIS 微内核; 朴素 attention; 最保守 CPU 路径 |

**节点级差异矩阵**:

| 节点 | sm_100+ | sm_90 | sm_80 | sm_70 | SME2 | AVX10.2+APX | AMX | AVX-512 |
|------|---------|-------|-------|-------|------|-------------|-----|---------|
| 权重加载 | FP4 直读 (无反量化) | FP16/BF16 pack | FP16 pack | FP16 pack | FP16 load | FP16/BF16 pack | BF16 tile load | FP32 pack |
| QKV GEMM | tcgen05.mma + block_scale | WGMMA 16×16×64 | mma.sync 16×8×16 | wmma 16×16×16 | SME2 outer product | BLIS VNNI-256 | AMX tdpbssd | BLIS FP32 |
| QK^T Scale | **无 (block-scaled 内置)** | Epilogue max | Epilogue max | Epilogue max | Epilogue max | Epilogue max | Epilogue max | Epilogue max |
| Softmax | Online + 2-CTA spec | Online + warp spec | Online tiled | Online tiled | Full (ZA 内) | Full | Full | Full |
| AV 计算 | tcgen05.mma + TMEM prefetch | WGMMA + TMA V | mma.sync | wmma | SME2 outer product | BLIS VNNI-256 | AMX tdpbssd | BLIS FP32 |
| FWHT 旋转 | Epilogue 内联 (GPU SIMD) | Epilogue 内联 | Epilogue 内联 | 可选跳过 | SVE2 内联 | AVX10 256-bit 内联 | AVX-512 内联 | AVX-512 内联 |
| KV 写入 | TMEMScatter (2-CTA) | Scatter 单 kernel | DtoD 逐页 | DtoD 逐页 | Direct write | Direct write | Direct write | Direct write |
| Sparse Mask | 无需 (block-scaled) | 可选 | 可选 | 无 | 无 | VP2INTERSECT 硬件 | 无 | 无 |

#### 13.12.2 FFN 子图拓扑

| 硬件 Profile | 子图拓扑 | 关键差异 |
|---|---|---|
| **sm_100+** | `NormIntoGemm → BlockScaled_Gate(tcgen05,FP4直读) → SiLU_Epilogue(dead_neuron) → GateSkip(2-CTA compress) → Up_GEMM(shared_pack,FP4) → FusedMulAdd → Down_GEMM(FP4)+Epilogue(residual)` | FP4 权重全程无反量化; block-scaled 消除 Gate/Up 独立 Scale 节点; 2-CTA compress 支持最高挤压比 |
| **sm_90** | `NormIntoGemm → Gate_GEMM(WGMMA) → SiLU_Epilogue → GateSkip(vcompress) → Up_GEMM(shared_pack_a) → FusedMulAdd → Down_GEMM+Epilogue` | WGMMA + warp spec; NormIntoGemm 融合; GateSkip 条件跳过 |
| **sm_80-89** | `RmsNorm → Gate_GEMM(mma.sync) → SiLU → GateSkip → Up_GEMM(shared_pack_a) → FusedMulAdd → Down_GEMM+Epilogue` | RmsNorm 独立节点 (寄存器约束); mma.sync; GateSkip warp prefix sum |
| **sm_70-79** | `RmsNorm → Gate_GEMM(wmma) → SiLU → Up_GEMM → Mul → Down_GEMM → Residual_Add` | 无 GateSkip (16 寄存器不够); 无 shared_pack_a; 所有节点独立 |
| **AVX10.2 + APX** | `RmsNorm → Gate_GEMM(BLIS,VNNI-256) → SiLU → GateSkip(vcompressps,31GPR) → Up_GEMM(shared_pack_a) → FusedMulAdd → Down_GEMM → Residual_Add` | 31 GPR 允许 SiLU→GateSkip→Up 全融合 (8-op epilogue 无溢出); vcompressps 硬件压缩; P-core 全力, E-core 标量降级 |
| **AVX-512 + AMX** | `RmsNorm → AMX_Gate_GEMM → SiLU → GateSkip(vcompressps) → AMX_Up_GEMM → Mul → AMX_Down_GEMM → Residual_Add` | AMX tile GEMM 替代 BLIS; GateSkip 可选; 32 zmm epilogue |
| **SME2** | `RmsNorm → SME2_Gate_GEMM(outer product) → SiLU → Up_GEMM → Mul → Down_GEMM → Residual_Add` | SME2 outer product 做 GEMM; ZA array 消除 pack/unpack 节点; 无 GateSkip (ZA 无 compress) |

**Gate-First Skip 拓扑条件** (§13.1):

| 硬件 | 支持 | 压缩机制 | 拓扑效果 |
|------|------|---------|---------|
| sm_100+ | ✅ tcgen05 + TMEM | 2-CTA 协同 compress (跨 SM 协调) | 最高挤压比: 死神经元 > 50% 时 Up/Down GEMM 规模减半 |
| sm_90 | ✅ WGMMA + 32 寄存器 | vcompress GPU intrinsic | GateSkip 内联: vcompress → 紧凑 Up/Down |
| sm_80-89 | ✅ mma.sync + 32 寄存器 | warp prefix sum | GateSkip 内联: prefix sum 挤压 |
| sm_70-79 | ❌ wmma + 16 寄存器 | — | **无 GateSkip 节点** |
| AVX10.2 + APX | ✅ vcompressps + 31 GPR | vcompressps 硬件指令 | 31 GPR 允许 GateSkip + 后续 GEMM 全融合无溢出 |
| AVX-512 + AMX | ✅ vcompressps | vcompressps | 可选，32 zmm 支持 |
| SME2 | ❌ | — | **无 GateSkip 节点**: ZA array 无 compress 操作 |

#### 13.12.3 KV Cache 写入子图拓扑

| 硬件 Profile | 子图拓扑 | 关键差异 |
|---|---|---|
| **sm_100+** | `RoPE → FWHT_K → Direct_FP4_K_Write(TMEM) → FWHT_V → Direct_FP4_V_Write(TMEM)` | FP4 权重环境: KV 可直接 4-bit 写入，**无独立 Quantize 节点**; TMEM 直写消除 scatter launch 开销 |
| **sm_90** | `RoPE → FWHT_K → Quantize_K → KvScatterWrite(per-head) → Quantize_V → KvScatterWrite(per-token)` | Scatter 单 kernel; 需要 Quantize 节点 (FP16→4-bit) |
| **sm_80-89** | `RoPE → FWHT_K → Quantize_K → DtoD_Copy(per_page) → Quantize_V → DtoD_Copy(per_page)` | 无 scatter; 逐页 DtoD |
| **sm_70-79** | `RoPE → FWHT_K → Quantize_K → DtoD_Copy → Quantize_V → DtoD_Copy` | 同 sm_80 但带宽更低 |
| **AVX10.2 + APX** | `RoPE(AVX10) → FWHT_K(256-bit SIMD) → Quantize_K(VNNI-256) → Direct_memcpy → Quantize_V → Direct_memcpy` | VNNI-256 加速量化; 31 GPR 允许 RoPE→FWHT→Quantize 三合一 epilogue |
| **AVX-512 + AMX** | `RoPE → FWHT_K → Quantize_K → Direct_memcpy → Quantize_V → Direct_memcpy` | AVX-512 量化; 直写内存 |
| **SME2** | `RoPE(SVE2) → FWHT_K(SVE2) → Quantize_K → Direct_memcpy → Quantize_V → Direct_memcpy` | SVE2 predicated 向量做量化; 无 scatter 概念 |

**sm_100+ FP4 颠覆性拓扑变化**:
- 传统路径 (sm_80-90): `FP16_K → Quantize_4bit → Write` (3 步)
- sm_100+: `FP4_K → Write` (1 步，权重已是 FP4，直接写入 KV cache)
- **Quantize 节点被完全消除**: 这是图拓扑的根本性变化，不是 codegen 差异

#### 13.12.4 融合模式硬件决策树

```
DeviceProfile
  │
  ├─ backend == Gpu?
  │   ├─ YES → sm_version?
  │   │   ├─ sm_100+ → FP4 直读(无反量化) + block_scaled(无 Scale 节点)
  │   │   │         + TMEM 替代 SMEM 做 attention tiling
  │   │   │         + tcgen05.mma(2-CTA 协同)
  │   │   │         + TMEMScatter_KV(无 scatter launch)
  │   │   │         + NormIntoGemm + GateSkip(2-CTA compress)
  │   │   │         + Quantize 节点全部消除(FP4 原生)
  │   │   │
  │   │   ├─ sm_90 → TMA_prefetch + WGMMA + warp_specialization
  │   │   │         + Scatter_KV + NormIntoGemm + GateSkip(vcompress)
  │   │   │         + Quantize(FP16→4bit) 节点保留
  │   │   │
  │   │   ├─ sm_80-89 → cp_async + mma.sync + DtoD_KV
  │   │   │             + NormIntoGemm(可选) + GateSkip(prefix sum)
  │   │   │             + Quantize 节点保留
  │   │   │
  │   │   └─ sm_70-79 → global_load + wmma + DtoD_KV
  │   │                 + Standalone + 无 GateSkip + FWHT 可跳过
  │   │                 + ComputeRoot(SMEM < 48KB)
  │   │
  │   └─ NO → CPU
  │       ├─ arch == AArch64?
  │       │   ├─ has_sme2? → SME2_tile_load + outer_product + Direct_KV
  │       │   │             + ZA_array 消除 pack/unpack
  │       │   │             + streaming mode 切换在层边界
  │       │   │             + 无 GateSkip (ZA 无 compress)
  │       │   │
  │       │   ├─ has_sve2? → SVE2_predicated_GEMM + Direct_KV
  │       │   │             + SVE2 epilogue (可变长)
  │       │   │
  │       │   └─ NEON → BLIS + NEON epilogue (最保守)
  │       │
  │       └─ arch == x86_64?
  │           ├─ avx10_version?
  │           │   ├─ AVX10.2 → 31 GPR + 256-bit unified
  │           │   │          + VP2INTERSECT 硬件 sparse mask
  │           │   │          + P/E hybrid: P-core 全 SIMD, E-core 标量降级
  │           │   │          + GateSkip(vcompressps, 31GPR 全融合无溢出)
  │           │   │          + VNNI-256 加速量化
  │           │   │
  │           │   └─ AVX10.1 → 31 GPR + 256-bit unified
  │           │              + 无 VP2INTERSECT (软件 sparse mask)
  │           │              + GateSkip(vcompressps)
  │           │
  │           ├─ has_amx? → AMX_tile + tdpbssd + Direct_KV
  │           │           + 32 zmm epilogue (8-op 无溢出)
  │           │           + GateSkip(vcompressps 可选)
  │           │
  │           ├─ has_avx512? → BLIS + 32 zmm epilogue
  │           │               + 无 AMX, 回退 BLIS 微内核
  │           │
  │           └─ AVX2 only → BLIS + 16 ymm epilogue
  │                       + 最保守: 溢出到栈, 限制 4-op epilogue
  │
  └─ turboquant_enabled?
      ├─ YES → FWHT 旋转节点:
      │        GPU: Epilogue 内联 (3 条 GPU SIMD)
      │        CPU x86: Epilogue 内联 (AVX-512 / AVX10 / AVX2)
      │        CPU ARM: Epilogue 内联 (SVE2 / NEON)
      │        sm_100+: FWHT 收益最大 (FP4 域内旋转, 无精度损失)
      └─ NO → 标准前向传播拓扑
```

**FusionRule 接口扩展**:

```rust
trait FusionRule {
    /// 匹配阶段: 检查子图模式是否适用 (不变)
    fn matches(&self, subgraph: &[OpNode]) -> Option<MatchResult>;

    /// 规划阶段: 根据 DeviceProfile 产生不同的图拓扑 (增强)
    fn plan(&self, match_result: MatchResult, profile: &DeviceProfile) -> FusionPlan {
        // 基于 profile 决定:
        // 1. 插入/消除节点 (sm_100+ 消除 Quantize; AVX10.2 消除软件 sparse mask)
        // 2. 选择计算核心 (tcgen05.mma / WGMMA / mma.sync / AMX / SME2 outer product)
        // 3. 连接边 (block_scaled 内置 scale / shared_pack_a / NormIntoGemm 直通)
        // 4. 融合深度 (31 GPR 8-op epilogue / 16 ymm 4-op / wmma standalone)
        // 5. 内存层级 (TMEM / SMEM / L1 / ZA array)
    }
}
```

**与 §12 空间异构的协同**:

§12 定义了"不同请求走不同子图"。本节定义了"同一请求，不同硬件走不同子图"。两者正交组合：
- §12 按请求形态 (Dense/Sparse/MoE) 分区 → 产生不同子批次
- §13.12 按硬件能力 (sm_100/sm_90/sm_80/AVX10.2/SME2/AMX/...) → 产生不同子图拓扑
- 最终: (请求形态 × 硬件能力) 的笛卡尔积由 §12.4 Shape Bucketing 统一调度

---

## §14 旧世代优化理念的全面突变升级 (ARCH-LEGACY-METAMORPHOSIS)

> **实现状态**: ✅ 已完成 — gllm 侧 §14.1-§14.5 + §15.1-§15.4 全量实现
> **关联**: 本文档 §9 Mega-Kernel, §13 Epilogue 白嫖
> **核心宣言**: 在"物理级隔离、指令级热修与 TurboQuant 降维"架构下，过去的优化思路发生了根本性突变。我们抛弃了所有 if-else 幼稚做法，将它们转为底层硬件法则。

### 14.1 动态混合精度检测 → 数学级静态湮灭 (Mathematical Annihilation)

- **旧思路**: RmsNorm 尾端计算 `Amax`，发现 Outlier 就回退 FP16，否则降级 FP8/INT8。反复横跳引发严重流水线不确定性。
- **新架构蜕变**: 详见 §11 TurboQuant 2.0 无损量化体系。
- **执行定论**: 所有 `Amax` 运行时检测代码**全盘删除**！QuantType 驱动 JIT 静态锁定执行路径。

### 14.2 门控网络失效截断 → 寄存器级动态挤压 (Register-Level Compaction)

- **旧思路**: 算完 `Gate = SiLU(xW)` 后发现死列，在调度层切片或 `jmp` 绕开。
- **新架构蜕变**: 在 Mega-Kernel 中，**不跳远，只挤压**。
- **执行定论**: 调用硬件谓词掩码（Predicated Execution）或张量挤压（AVX512 `vcompress`、GPU `Prefix Sum`）。死掉的神经元在 SMEM 里挤成最小密实体，`Up/Down GEMM` 全负荷不停机算平。

### 14.3 跨层残差旁路 → 块级内嵌路由自毁弃算 (In-Kernel Router Veto)

- **旧思路**: Residual Add 算方差 $\Delta \rho < 0.001$ 后走控制流分支短路。
- **架构校正（大并发禁区）**: 绝对不能用宏观热修补！因为 Continuous Batching 下 Request A 可以跳过，但 Request B 正处于高信息熵刚需期。修改汇编 `jmp` 会导致 B 产出乱码！
- **执行定论**: 残差跳过归入 **§9.1 Mega-Kernel 块级内嵌路由**：
  - Request A 的 Thread Block 读取到 $\Delta \rho < 0.001$ → 触发 `Thread Exit`，输入原封不动抛给输出
  - Request B 的 Thread Block 读取到 $\Delta \rho = 0.5$ → 正常展开矩阵乘法循环
  - 物理同源、逻辑分块（Block Idx）的掩码路由，不破坏全局机器码

### 14.4 Hot JMP Patching 的正确使用场景

既然个体行为不能导致热修补，那热修补**仅用于绝对的全局物理共识**：
- **场景 A（领域降维）**: MoE 冷板凳专家持续数百万 Token 均 0 命中率 → JIT Director 用 `NOP/JMP` 物理抹平其访存分支和跳转入口（DCE）
- **场景 B（极致前缀复用）**: 128 个并发请求全部挂载同一个 10K System Prompt 前缀树 → 前置 Prefill 图直接被热修补塌缩为一条共享直读存储指令

### 14.5 注意力预瞄 → RDMA/PCIe 流水线预取 (Pipelined Prefetch)

- **旧思路**: 用 Softmax 质心决定是否算全量 `Q*K^T`。
- **新架构蜕变**: 将"算力的减法"转为"带宽的乘法"。
- **执行定论**: Softmax 质心坐标通过页表直接馈入底层硬件预取系统。GPU 处理本层 Dense 网络的同时，总线通道上已在并行利用 `cuMemPrefetchAsync` 甚至跨机 `RDMA` 加载下一层的 KV 块。**彻底穿透冯诺依曼架构的显存墙诅咒！**

---

## §15 MoE 异构专家极致落地详案 (ARCH-MOE-EXTREME)

> **实现状态**: ✅ IMPLEMENTED — gllm 侧 MoE 基础设施已完成 (routing/thermal/prefetch/dispatch/hot_patch/prefetch_pipeline), gllm-kernels JIT codegen 待集成
> **关联**: 本文档 §9 Mega-Kernel + §15 MoE
> **核心宣言**: 在"大一统 JIT + TurboQuant + Deopt 兜底"的底层生态下，MoE 的异构不再是孤岛功能，而是底层组件相互摩擦后的自然现象。

### 15.1 核内分发与零启动开销 (In-Kernel Expert Dispatch)

传统 MoE 引擎（如 vLLM/TGI）为每个 Expert 调用独立 Kernel，导致严重 Launch Overhead。
- MoE 层依然只启动 **1 个 Mega-Kernel**
- Thread Block 拿到 `Softmax(Gate)` 后利用内置字典读出目标 Expert
- Kernel 内部利用汇编 `jmp` 直接跃迁到对应专家的权重读取区
- 物理空间上并发解耦，**零 Driver 启动开销**

### 15.2 TurboQuant + 预瞄 → Zero-Stall Swapping

温/冷专家权重在 CPU RAM 甚至远端 RDMA 内存中时：
- TurboQuant 2.0 将专家权重暴压到 4-bit 甚至 2-bit，**PCIe/RDMA 搬运延迟缩减 75%~87.5%**
- Gate 层算出路由表的瞬间，立刻启动 `cuMemPrefetchAsync` 无阻塞预加载
- Thread Block 走到该 Expert 汇编入口时，被极度压缩的权重**已躺在 GPU L2 Cache 里**
- 冷专家卡顿被计算流水线完美掩盖（Pipelining）

### 15.3 CPU/GPU 真正并行 (Core Disaggregation)

- CPU 的 NUMA 探针暴露 `AMX` 或 `AVX512` 能力为 IR
- 编译器为温专家顺量生成一段 CPU 特化的 JIT 代码
- 通过独立的并发流将少部分 Token 交给 CPU 算完后统一回写
- 不需要上层写多线程锁

### 15.4 冷板凳专家的全域封杀与"复活陷阱" (Uncommon Traps & OSR Bailout)

这是所有激进 JIT 编译器（如 V8 / Java HotSpot）必须解决的最核心难点：**物理截肢了一个专家后突然又需要它，怎么办？**

1. **绝对正确性底线**: 门控（Gate Router）计算开销极低（< 1% 总算力），**绝不去热修补 Gate 概率计算本身**
2. **陷阱替换 (Uncommon Trap)**: JIT Director 抹除冷门专家的 Up/Down GEMM 时，不是无脑 `NOP` 丢弃，而是覆写为指向 **Deopt Handler** 的跳转指令
3. **冷触发复活流程 (OSR Bailout)**:
   - 过去几百万次运行，无人需要冷专家，系统高速穿透修补后的平滑防线
   - 突然 Request A 的 Gate 指向被封杀的冷专家 7
   - Request A 的 Thread Block 一头撞进 **Uncommon Trap**
   - 该 Thread Block 立刻向显存写下 `DEOPT_REQUEST = 7` 并主动挂起（不输出错误数据）
   - 同 Batch 的其他常规请求不受影响，继续极速算完
   - 这一层结算时，引擎主循环发现 `DEOPT_REQUEST`
   - Host 触发不到 **1ms 的微冷冻**，JIT Director 瞬间回写 `.text` 复原冷专家 7 的网格
   - 异步唤回其在主存的 4-bit 权重
   - 引擎为挂起的 Request A 单独走一遍回炉重造（Re-evaluate）

> 这种 **"用万分之一的局部挂起代价，换走 99.99% 时间里的物理绝对零开销"** 的设计，就是 De-optimization（去优化回退）机制，构成 Gllm 架构的最强韧防御底盘。

---

## §16 残差总线的四大物理应用全景 (ARCH-RESIDUAL-BUS-APPLICATIONS)

> **实现状态**: ✅ IMPLEMENTED — rag.rs(88L/✅ Late-Fusion RAG), early_exit.rs(759L/✅ ConfidenceCalibrator+AdaptiveExitPoints+ExitLayerStats+EarlyExitController), guardrail.rs(776L/✅ GuardProbeRunner+GuardVetoState), intent.rs(730L/✅ IntentEncoder+IntentClassifier+IntentProbe), routing.rs(467L/✅ ResidualBus port management), executor.rs(✅ ResidualBus+EarlyExitController+per-request lifecycle+encode_intent delegation)
> **核心拷问**: 残差连接 $x_{out} = x_{in} + \text{Layer}(x_{in})$ 不仅仅是为了梯度不消失。在 JIT Mega-Kernel 架构中，残差流被物理重构为一条**贯穿始终的开放式数据总线**。

### 16.1 超大知识的外挂注入点 (Late-Fusion RAG Injection)

- 外部知识（如 10 万字财报）**不需要从第 0 层 Embedding 开始爬**
- 用极便宜的小模型（BERT / 2B）预算出高维语义向量
- JIT 在指定的"知识融合层"（如第 15 层）预留外链读取汇编（`LDG.E` 外部图显存）
- 系统拉开该层残差入口，用极其极速的 `Vector Add` 指令直接将知识向量**硬加进残差流**
- 外部长上下文物理级跳过前半段网络，直接在深层语义区完成**晚期融合（Late-Fusion）**

### 16.2 任意层数据召回与高维截断 (PGSLE Early-Exit)

- 在 Mega-Kernel 生成时，JIT 在特定层（如 Layer 20, 28, 35）的残差 Add 后顺手附带一个**超小的线性分类器（微型 lm_head）**
- 如果在第 20 层，微型分类器"召回"的中间数据算出下一个 Token 概率逼近 **99.9%**（如模型正在复颂成语"大海(捞针)"）
- 该层 Thread Block 直接发射控制信标，触发 §9.1 块级路由或 §9.2 热修补，**当场物理切断后续 20 层计算**
- 对于简单问题的运算量被**物理腰斩**

### 16.3 纯解码降维：通用多意图识别 (Pure_Decode Intent NLU)

- 大模型本身就是无可匹敌的语义编码器
- 引擎提供 `Pure_Decode` API 模式：JIT 编译时仅选取前 15-20 层"理解区"（剥离后续语言生成层）
- Prompt 跑过这前段图后，JIT 直接把残差总线截留（Recall），送入轻量级多分类探针（Linear Probe）
- 引擎化身为**速度媲美小模型、成本近乎 0 的意图分类与特征提取神塔**

### 16.4 零延迟飞行巡航审查 (In-Flight Guardrail Layer)

- 不再是输入前拦截或输出后审查的二次判断
- 在模型深度的某一层（如倒数几层）**物理强插入极小的安全审查头（Safety Head）**
- 自回归生成过程中，每吐出一个新词时，寄生在残差总线上的 Guard 探针不断"嗅探"当前高维语义流中是否存在危险概念特征聚集
- **物理熔断**: 一旦 Guard 探针概率超标，Mega-Kernel 内的 Thread Block 直接抛出 `Safety Veto` 中断信号
- 模型在吐出危险词的**那一瞬间之前**，就被当场强行切断计算流
- 这是真正的**零延迟、无法越狱的安全物理护城河**

> **架构结论**: 诸如意图识别、RAG、安全护栏等原先必须架设微服务集群的复杂大生态，被硬生生压成了 Kernel 里的几句附加指令，化为引擎出厂自带的超光速内置兵器！

---

## §17 JIT 自适应推测解码 — 硬件指令级并发与精度并重 (ARCH-ADAPTIVE-SPEC)

> **实现状态**: ✅ IMPLEMENTED — 推测解码核心模块已实现 (adapter/tree/verify/cache/engine), 35 单元测试全部通过
> **学术依据**: EESD (ACL 2024), ADEPT (arXiv:2601.03700), Goose (arXiv:2604.02047), EqSpec (arXiv:2510.22876), TIDE (arXiv:2603.21365)
> **关联**: §9 Mega-Kernel 块级路由 (Compact→Execute→Scatter 三段式), §10 Chunked Prefill, §16.2 PGSLE Early-Exit, §11 TurboQuant
> **核心拷问**: gllm 已有 PolymorphicExecutor 多变体架构（§16）和 Compact→Execute→Scatter（§9.1）。§17 将两者物理级对接，用模型自身的浅层变体充当 Draft Model，配合硬件指令级 batch 合并和各向异性推测树，在**零额外模型权重**的前提下将单步 decode 吞吐提升 2-3×。

### 17.1 自推测核心管线 (Self-Speculative Decoding)

**核心洞察**: PolymorphicExecutor 已预编译了覆盖不同层数的变体（L2_hot ≈ L/3 层, L3_warm ≈ 2L/3 层, full = L 层）。最浅变体天然充当 Draft Model — 无需任何额外权重。

```
┌────────────────────────────────────────────────────────────────┐
│                    Executor::step_speculative()                │
│                                                                │
│  Phase A: Draft (L2_hot variant)                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Preamble(1×) → Variant[L2_hot](K layers) → Adapter      │  │
│  │                                ↓                          │  │
│  │                    Anisotropic Spec Tree                  │  │
│  │              (PLD spine + n-gram branches)                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓ tree_tokens                         │
│  Phase B: Verify (full variant) — 单次 batched forward         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Preamble(1×) → Variant[full](L layers) + Tree Attn Mask  │  │
│  │                                ↓                          │  │
│  │              EqSpec: per-sequence verify + atomic commit  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓ accepted_tokens                     │
│  Phase C: Shadow KV Fill (ADEPT)                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 低秩投影填充 draft tokens 在 K+1..L-1 层的 KV 缺口       │  │
│  │ ShadowProjector: JIT 编译的 [hidden→2*kv_dim] 投影图      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Phase D: SIMD/Warp 级 Compact→Execute→Scatter                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 异构 accepted_length → 物理挤压 → 稠密计算 → 原位散射回写  │  │
│  │ CPU: AVX-512 vcompressps | GPU: Warp Vote + Prefix Sum    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

**关键不变量**:
- **I1**: Draft 和 Verify 共享同一份 Preamble 输出（embedding 只计算一次）
- **I2**: Adapter 权重 = `lm_head.weight`（零额外参数，Phase A 方案）
- **I3**: Verify phase 对所有 tree tokens 执行完整 L 层前向，自然产生完整 KV — **无需 Shadow KV 填充**
- **I4**: Shadow KV 仅用于 §16.2 PGSLE Early-Exit 场景（tokens 跳过深层），不用于推测解码

### 17.2 Adapter — 零参数 Draft 投影头

Adapter 是附在 L2_hot 变体末尾的微型投影图，将中间 hidden state 映射到 vocab 空间。

**图结构** (JIT 编译为独立 FusedGraph):
```
hidden_state[K, hidden_size]
  → RmsNorm(hidden_size, eps=1e-5)     // 复用最后一层的 norm 权重
  → MatMul(normed, lm_head.weight^T)   // 复用 lm_head.weight, 零额外参数
  → logits[vocab_size]
```

**JIT 编译集成**:
- Phase 0 Scalar: `y = (x / rms) * weight`, `logits = normed @ W^T`
- Phase 1 SemanticDAG: 自动分类为 `ElemWise(RmsNorm) + Gemm(MatMul)`
- Phase 2 Fusion: `NormIntoGemm` — RmsNorm 输出直喂 GEMM（无中间写回，符合 §5 Deep Fusion 设计）
- Phase 3 ISA Lowering:
  - x86_64: AVX-512 `vrsqrt14ps` + `vfmadd231ps` + BLIS GEMM
  - AArch64: SVE `frsqrte` + FMLA + micro-kernel
  - GPU: Tensor Core GEMM with RmsNorm epilogue injection

**编译时机**: `from_loader()` 阶段，与 PolymorphicExecutor 的变体编译同时完成。

**自适应 Adapter 权重 (可选)**:
```
Phase A (零参数): adapter_weight = lm_head.weight (共享, 无额外内存)
Phase B (微调):   adapter_weight = lm_head.weight + residual_delta
                  residual_delta 在前 100 步用 full model 的真实 logits 蒸馏
                  额外参数: vocab_size × hidden_size (≈0.1% 总模型大小)
```

### 17.3 各向异性推测树 (Anisotropic Speculation Tree)

传统推测解码使用均匀等深树（isotropic tree），所有分支深度相同。但真实 token 接受率高度异构 — 高频词接受率 >80%，生僻词 <10%。Goose (arXiv:2604.02047) 证明各向异性拓扑的理论最优性。

**树结构定义**:

```
SpecTree:
  ┌─ spine[0] (top-1 from adapter, acceptance ≈ 70%)
  │   ├─ spine[1] (PLD continuation from prompt, acceptance ≈ 60%)
  │   │   ├─ spine[2] (PLD continuation, acceptance ≈ 55%)
  │   │   │   ├─ spine[3] (PLD, acceptance ≈ 50%)
  │   │   │   │   └─ spine[4] (PLD, acceptance ≈ 45%)
  │   │   │   ├─ branch[2,0] (n-gram alt-1)
  │   │   │   └─ branch[2,1] (n-gram alt-2)
  │   │   ├─ branch[1,0] (n-gram alt-1)
  │   │   └─ branch[1,1] (n-gram alt-2)
  │   └─ branch[0,0] (adapter top-2, acceptance ≈ 15%)
  └─ branch[root,0] (adapter top-3, acceptance ≈ 5%)
```

**两个训练无关的 Draft 来源**:
1. **PLD (Prompt Lookup Decoding)**: 在 prompt + 已生成 tokens 中匹配 n-gram，延续为 spine 候选。接受率 60-80%（任务有重复模式时）。
2. **N-gram Recurrence**: 从 prompt 频率表提取 top-k 替代候选，作为 branch 节点。接受率 5-15%，但提供多样性。

**非退化保证** (Goose Proposition 1): 组合 PLD+TR 树的 accepted tokens ≥ 任一单源方法。数学证明：组合树是两棵独立树的超集。

**Tree Attention Mask 构建**:
- 每个 tree 节点只 attend 到其 root→node 路径上的 token（因果约束）
- Mask 是稀疏二值矩阵，shape = [tree_size, total_seq + tree_size]
- 传入 `run_with_tree_mask()` 作为 JIT 运行时参数（ShapeBinding）

### 17.4 EqSpec — Batch 正确性三不变量

Speculative Decoding 的 batch 化存在隐蔽的正确性陷阱 (arXiv:2510.22876)。gllm 强制执行三条不变量：

**I1 (Topology Invariant)**: 同一 batch 内所有 sequence 共享相同的 SpecTree 拓扑。各 sequence 的 accepted_length 可以不同。

**I2 (Single Verification)**: 所有 sequence 的 tree verification 在**一次** batched forward 中完成。每个 sequence 有独立的 tree attention mask。

**I3 (Atomic KV Commit)**:
```
for each sequence in batch:
    accepted_len = verify_prefix(target_logits, draft_tokens)
    // 只 commit accepted tokens 的 KV entries
    kv_cache.commit(sequence_id, accepted_tokens)
    // Rejected tokens 的 KV pages 立即回收到 free pool
    kv_cache.rollback(sequence_id, rejected_tokens)
```

**跨 sequence 隔离**: Sequence A 接受 7 tokens、Sequence B 接受 2 tokens 时，两者互不干扰。每个 sequence 的 KV commit/rollback 独立原子执行。

### 17.5 ADEPT 阴影 KV 填充 — Early-Exit 场景专用

> **适用范围**: 本节仅用于 §16.2 PGSLE Early-Exit 场景（token 跳过深层时产生 KV 缺口）。
> 推测解码场景中 Verify phase 已执行完整 L 层，KV 自然完整，**不需要** Shadow KV。

**问题**: 当 token 在第 K 层 early exit 时，KV cache 中第 K+1..L-1 层没有该 token 的 KV entries。后续 token 通过 attention 访问该位置时，缺失的 KV entries 导致注意力计算错误。

**ADEPT 方案**: 低秩投影近似填充缺失 KV。

```
Shadow Projector Graph (JIT compiled):
  exit_hidden[K, hidden_size]
    → Linear(hidden_size, rank=64, W_down)    // 降维: hidden → rank
    → Linear(rank, 2 * kv_dim, W_up)          // 升维: rank → K+V
    → KV[exit_layer+1..L-1]                    // 写入缺口层
```

**参数量**: `2 * hidden_size * 64 + 2 * 64 * num_kv_heads * head_dim` ≈ 总模型 0.05%

**精度保证**: rank≥64 时 perplexity 下降 < 0.1 (ADEPT 实验验证)。Shadow KV 是近似的，但注意力机制对深层 KV 的敏感度远低于浅层，近似误差被自然抑制。

**在线校准**: 前 100 步用 full model 的真实 KV 与 shadow KV 的 L2 距离作为蒸馏信号，微调 W_down 和 W_up。

### 17.6 硬件指令级 Batch 合并 — Compact→Execute→Scatter 的 SIMD/Warp 实现

§9.1 定义了 Compact→Execute→Scatter 三段式范式。§17.6 给出该范式在推测解码异构 batch 场景下的具体硬件实现。

#### 17.6.1 CPU SIMD Lane Compaction (x86_64)

当 batch 中不同 sequence 有不同的 `accepted_length`（如 seq_0 接受 5 tokens, seq_1 接受 2 tokens），后续 KV commit 操作需要处理异构长度的数据。

**AVX-512 物理挤压**:
```asm
; 假设 8 个 sequence 在 ZMM0-ZMM7 中有 hidden states
; active_mask = 每个 sequence 是否有剩余待处理 tokens
; Step 1: Compact — 将 active lanes 挤压到连续位置
vptestmd k1, zmm_active, zmm_all_ones   ; 生成 opmask
vpcompressd zmm_compact{k1}, zmm_data    ; 物理挤压 active elements
; Step 2: Execute — 在稠密数据上执行 GEMM (全部 SIMD lanes 有用)
vfmadd231ps zmm_acc, zmm_compact, zmm_weight
; Step 3: Scatter — 按 original lane offset 写回
vpscatterdd zmm_output{k1}, [base + zmm_indices*4]
```

**性能收益**: 消除 wasted SIMD lanes。8-wide AVX-512 中，如果只有 5/8 序列活跃，未挤压时浪费 37.5% 算力；挤压后 100% 利用。

#### 17.6.2 GPU Warp-Level Compaction

**CUDA/Warp 级挤压**:
```cuda
// Step 1: Warp Vote — 统计 active threads
int active_mask = __ballot_sync(0xFFFFFFFF, has_remaining_tokens);
int active_count = __popc(active_mask);
// Step 2: Prefix Sum — 计算每个 active thread 的压缩位置
int compact_pos = __popc(active_mask & ((1 << lane_id) - 1));
// Step 3: Exchange — 挤压到 warp 前部
T compacted_data[32];
if (has_remaining_tokens) {
    compacted_data[compact_pos] = my_data;
}
// Step 4: Execute — GEMM on compacted data (全部 32 线程有用)
gemm_kernel(compacted_data, ...);
// Step 5: Scatter — 按 original lane_id 写回
if (has_remaining_tokens) {
    output[original_offset + lane_id] = compacted_data[compact_pos];
}
```

**Tensor Core 兼容**: Compact 后数据连续排列，可直接 feed `wmma::load_matrix_sync` / `mma.sync`。无 padding bubbles。

#### 17.6.3 AArch64 SVE Predicate Compaction

```asm
// SVE 的 predicate register 天然支持 sparse lane 操作
whilelt p0.s, x_idx, x_count    ; 生成 active lane mask
ld1w z0.s, p0/z, [x_base]       ; 仅加载 active lanes (zeros inactive)
// 无需显式 compact — predicate 寄存器自动 mask
fmla z_acc.s, p0/m, z0.s, z_w.s ; 仅 active lanes 参与 FMA
st1w z_acc.s, p0, [x_out]       ; 仅写回 active lanes
```

**关键区别**: SVE 不需要显式 compact。Predicate 寄存器 (`p0`) 在加载、计算、存储三阶段自动过滤 inactive lanes。这是 SVE 相对 AVX-512 的架构优势。

### 17.7 稀疏 KV Cache 布局与内部布局代价模型

#### 17.7.1 Early-Exit 产生的 KV 空洞

Early-Exit token 的 KV cache 物理形态:

```
Layer:   0   1   2  ...  K-1   K   K+1  ...  L-1
Token 0: [KV][KV][KV]...[KV]  [KV][KV] ... [KV]  ← 完整 (全层通过)
Token 1: [KV][KV][KV]...[KV]  [  ][  ] ... [  ]  ← 缺口 (在层 K exit)
Token 2: [KV][KV]     ...[  ] [  ][  ] ... [  ]  ← 缺口 (在层 2 exit)
```

**三种布局策略**:

| 策略 | 物理表示 | 适用场景 | 代价 |
|------|---------|---------|------|
| **Dense+Shadow (推荐)** | 全量 KV 分配，缺口层由 ADEPT Shadow 填充 | 推测解码 + Early-Exit 混合 | 额外 ~0.05% 参数，零运行时分支 |
| **Per-Layer Bitmap** | 每层一个 bitmap 标记哪些 token 有 KV entry | 纯 Early-Exit，无 shadow | Attention 需要 mask 无效位置，~3% 额外访存 |
| **CSR Compressed** | 层→token 映射用 CSR 格式 | 极端稀疏 (>80% tokens early-exit) | 复杂度高，间接寻址开销 |

**推荐 Dense+Shadow**:
- 保持所有 Attention kernel 的 dense 布局不变（零修改）
- Shadow KV 填充在 KV Write 阶段完成（白嫖 §9.5 Epilogue 尾段）
- 硬件级零分支 — Attention 计算路径无任何条件判断

#### 17.7.2 内部布局代价模型 (Layout Pricing Model)

JIT 编译器在融合决策时需要量化不同数据布局的硬件代价。

**代价公式**:
```
LayoutCost(layout, access_pattern, hardware) =
    CacheLineWaste(layout, access_pattern) × L1_latency
  + TLBPressure(layout) × TLB_miss_penalty
  + BankConflict(layout, simd_width) × conflict_stall_cycles
  + SIMDLaneWaste(layout, batch_heterogeneity) × wasted_flops_ratio
```

**四个维度**:

1. **CacheLineWaste**: 一次 cache line 加载中实际被使用的字节比例。
   ```
   waste = 1 - (useful_bytes / cache_line_bytes)
   Dense dense: waste = 0%
   Interleaved K/V: waste = 0% (K 和 V 交替排列，连续访问)
   Separated K|V: waste 取决于 seq_len 是否填满 page
   ```

2. **TLBPressure**: 数据布局导致的 page 表项数量。
   ```
   Paged (page_size=64 tokens): TLB entries = total_pages
   Dense per-sequence: TLB entries = num_sequences × num_layers
   ```

3. **BankConflict**: Shared Memory / L1 bank 冲突导致的 stall 周期。
   ```
   Head-major layout + 32 heads → 每 head 偏移 = head_dim × sizeof(f32) = 128 bytes
   128 bytes / 4 bytes per bank = 32 banks → 完美无冲突
   ```

4. **SIMDLaneWaste**: 因 batch 异构性导致的 SIMD lane 空闲率。
   ```
   无 compaction: waste = (batch_size - active_count) / batch_size
   有 compaction: waste = 0% (但 compact 本身有 ~2-3 cycle/lane 开销)
   自适应阈值: 当 waste > 25% 时触发 compact, 否则跳过 (compact 开销 > 浪费)
   ```

**JIT 编译器使用**: Fusion Engine (Phase 2) 在 `NormIntoGemm` vs `Standalone` 决策时，用 `LayoutCost` 评估中间张量写回的代价，选择总代价最小的融合路径。

#### 17.7.3 Early-Exit 检测与信号传播

> **关联**: §13.3 (Residual Bypass 融合), §9.5 (Epilogue Paged Telemetry), §16.2 (PGSLE Early-Exit)

**检测点**: §13.3 Residual Bypass 融合点——每层 RmsNorm 后的 Epilogue 尾段，白嫖计算两个指标：
- `cosine_sim`: 当前层残差与前一层的余弦相似度
- `delta_rho`: 残差能量比 `||x_out|| / ||x_in||`

**信号生成**:
```
ExitSignal {
    layer: usize,           // exit 层号 (0-based)
    token_id: TokenId,      // 被 exit 的 token
    confidence: f32,        // cosine_sim (≥ 0.99)
    delta_rho: f32,         // 能量比 (|delta - 1.0| < 0.01)
}
```

**信号传播**: ExitSignal 写入 §9.5 Epilogue Paged Header（零额外内存搬移）。后台 JIT Director Daemon 低频扫描 Paged Headers 收集信号。

**Mega-Kernel 内部处理**:
1. Thread Block 在层 K 的 Epilogue 尾段检测到 exit 条件
2. 标记当前 token 的 `exit_layer = K`（写入 token metadata）
3. 后续 K+1..L-1 层：Thread Block 跳过该 token 的计算（predicate mask 或 compact 移除）
4. Batch 结束后，KV cache manager 统计所有 early-exit tokens 的 exit_layer 分布

#### 17.7.4 稀疏 KV 策略选择算法

> **关联 REQ**: REQ-SPEC-015 (稀疏 KV 策略选择), 03-DATA-STRUCTURE.md §16.7 (`SparseKvContext`)

```
fn select_sparse_strategy(context: &SparseKvContext) -> SparseKvStrategy {
    let sparsity = context.hole_count as f64 / context.total_kv_entries as f64;
    let has_speculative = context.active_spec_requests > 0;

    // 约束: 推测解码 verify phase 自然产生完整 KV → 必须 Dense
    if has_speculative { return DenseShadow; }

    // 极端稀疏 (>80% tokens early-exit) → CSR 最省内存
    if sparsity > 0.80 { return CsrCompressed; }

    // 中等稀疏 → Per-Layer Bitmap (attention mask 开销可控)
    if sparsity > 0.20 { return PerLayerBitmap; }

    // 低稀疏 → DenseShadow (推荐，零运行时分支)
    return DenseShadow;
}
```

**策略切换规则**:
- 评估时机：每次 batch 前重新计算 `SparseKvContext`
- DenseShadow ↔ PerLayerBitmap：可渐进切换（两者共享全量 page 预分配，只是 bitmap 使能不同）
- 切换到 CSR：需要重建 CSR 索引（开销较大，仅在 sparsity 持续 > 80% 时触发）
- 从 CSR 切回：CSR → DenseShadow/PerLayerBitmap 需要重建全量 page（一次性开销）

#### 17.7.5 ShadowProjector 执行生命周期

> **关联 REQ**: REQ-SPEC-016 (ShadowProjector 生命周期), §17.5 (ADEPT 阴影 KV 填充)

**Phase 1: 编译**（模型加载时，与变体编译同时）
```
ShadowProjectorConfig { hidden_size, rank=64, kv_dim, calibration_steps=100 }
  → build_shadow_projector_graph()
    Linear(hidden_size, rank, W_down)    // 降维: hidden → rank
    + Linear(rank, 2 * kv_dim, W_up)    // 升维: rank → K + V
  → JIT 编译为 FusedGraphExecutor
  → 存入 ModelJitCache（与 Adapter 共享 ArtifactCache）
  → W_down, W_up: Xavier 随机初始化
```

**Phase 2: 运行时填充**（每次 batch 后）
```
for each early-exit token in batch:
    1. exit_hidden = hidden_states[token_id, :hidden_size]   // 从 exit 层截取
    2. shadow_kv = projector.forward(exit_hidden)            // [hidden→rank→2*kv_dim]
    3. kv_cache.write_range(token_id, exit_layer+1..L-1, shadow_kv)
       // 写入已分配的 KV pages（DenseShadow 全量预分配了所有层的 page 空间）
       // 不触发新 page 分配，不影响 page ref-count
```

**Phase 3: 在线校准**（前 `calibration_steps` 步）
```
// 混合模式：部分 token 正常通过全部层（ground truth），部分 early-exit
// 对每个 early-exit token：
//   1. 收集 full model 在 exit_layer+1..L-1 层的真实 KV (ground truth)
//   2. 收集 ShadowProjector 生成的 shadow KV
//   3. 计算 L2 loss: ||full_kv - shadow_kv||²
//   4. 累积 loss，每 calibration_batch_size 步反向传播
// 收敛条件: perplexity_delta < target_perplexity_delta (默认 0.1)
// 更新方式: SGD 微调 W_down, W_up
```

**ShadowProjector 与 PagedAttention 交互**:
- 填充目标：已分配的 KV pages（token 已有前 K 层 KV，后 L-K-1 层由 shadow 填充）
- 不触发新 page 分配（DenseShadow 全量预分配了所有层的 page 空间）
- Page ref-count 不受影响（同一 token 的 page 只是填充了 shadow 数据）
- 写入路径：与正常 KV write 共享同一 `write_kv_to_cache` 路径

#### 17.7.6 Per-Layer Bitmap 执行流程

**Bitmap 存储**:
- 每 page 一个 `u64` bitmap（对应 page 内 64 个 token slots，1 bit/token）
- `bit[i] = 1` 表示该 page 内第 i 个 token 在当前层有有效 KV entry
- 总开销：`num_layers × num_pages × 8` bytes（对 Llama-3-8B: 32 layers × ~1000 pages × 8B ≈ 256KB）

**Attention 消费**:
1. Attention kernel 加载当前层的 bitmap
2. 生成 attention mask：`bit[i] == 0` 的位置设为 `-inf`（等效于 attention weight = 0）
3. Softmax 自然忽略这些位置
4. 无需特殊 kernel 修改（仅 mask 生成逻辑不同）

**更新时机**:
- 新 KV entry 写入时：`bitmap[page_idx] |= (1 << token_offset_in_page)`
- Speculative rollback 时：`bitmap[page_idx] &= !(1 << token_offset_in_page)`
- DenseShadow 模式下 bitmap 全为 1（无更新开销）

**适用场景**: 纯 Early-Exit、无 shadow projector、中等稀疏度 (20%-80%)

#### 17.7.7 CSR Compressed 执行流程

**构建时机**: batch 结束后，由 KV cache manager 统计有效 entries 构建

**结构**:
```
CSR Sparse KV:
  indptr:  [0, valid_L0, valid_L0+valid_L1, ..., total_valid]  // (L+1) 个元素
  indices: [token_ids with valid KV in layer 0, layer 1, ...]   // total_valid 个元素
```

**Attention 消费**:
1. 通过 `indptr[layer_id]` 定位当前层的有效 token 范围 `[start, end)`
2. `indices[start..end]` 为有效 token IDs
3. Gather K/V: 按 indices 间接寻址获取 K/V 向量
4. 计算 attention（仅对有效 token 计算 QK^T）

**增量更新**:
- 新 early-exit token：添加到对应层的 indices 尾部，更新 indptr
- 全量通过 token：在所有层都添加到 indices
- 重建触发条件：indices 碎片率 > 30% 时全量重建

**限制**:
- Attention kernel 需要 CSR 特化路径（间接寻址，不适合 SIMD 向量化）
- 适合 GPU（warp-level gather 高效），不适合 CPU SIMD（gather 开销高）
- 仅在极端稀疏 (>80%) 时收益大于开销

#### 17.7.8 Sparse KV + PagedAttention 交互规则

> **关联 REQ**: REQ-SPEC-017 (Sparse KV + PagedAttention 交互)

| 策略 | Page 预分配 | Page 填充 | Page 管理 |
|------|------------|----------|----------|
| DenseShadow | 全量（所有层所有 token） | 前 K 层真实 KV + 后 L-K-1 层 shadow KV | 标准 PagedAttention，零变化 |
| PerLayerBitmap | 全量（同上） | 仅有效 token 位置写入 KV，bitmap 标记有效性 | 标准 PagedAttention + bitmap 更新 |
| CsrCompressed | 仅有效 entries（sparse page allocator） | 仅分配有效层 token 的 page | 需要 sparse page allocator |

**Page 共享（Prefix）**:
- 同一前缀的不同 sequence 如果 exit 层序列**完全一致**，可以共享前缀 KV pages
- exit 层不一致时不能共享（DenseShadow 模式下 shadow KV 填充质量因 exit 层不同而异）
- 共享判断：trie 节点比较 `exit_layers: Vec<u32>` 作为额外匹配条件

**Page 回收（Speculative Rollback）**:
- Rejected tokens 的 KV pages 整页回收
- Rollback 时清除 PerLayerBitmap 对应 bit
- CSR 模式下从 indices 中移除 rejected tokens

#### 17.7.3 推测解码的 KV Cache 布局优化

**Tree Token KV 存储策略**:
```
Verify phase 为 tree 中所有 N 个节点计算完整 KV:
  KV shape: [L layers, N tree_nodes, num_kv_heads, head_dim]

Accepted tokens 的 KV commit:
  - 复制 accepted 范围内的 KV 到主 KV cache 对应位置
  - Rejected tokens 的 KV 丢弃 (内存回收)
  - Commit 操作通过 DtoD (GPU) 或 memcpy (CPU) 完成，无需重计算

关键优化: tree KV 使用临时 buffer，不直接写入主 KV cache
  → 避免 rejected tokens 污染主 cache
  → 只有 accepted tokens 最终 commit
  → 临时 buffer 大小 = max_tree_size × L × 2 × kv_dim × dtype_size
    (Llama-3-8B: 5 × 32 × 2 × 1024 × 2 = 640KB, 可完全放入 L2)
```

### 17.8 编译时产物清单

模型加载时一次性编译的全部 JIT 产物:

| 产物 | 图结构 | 用途 | 参数来源 |
|------|-------|------|---------|
| **Preamble** | embed_tokens → hidden | 所有 phase 共享的 embedding | 原始权重 |
| **L2_hot Variant** | layers 0..K | Draft phase 主干 | 原始权重 |
| **L3_warm Variant** | layers 0..2K/3 | 中间变体 (备选) | 原始权重 |
| **Full Variant** | layers 0..L | Verify phase 主干 | 原始权重 |
| **Postamble** | final_norm + lm_head | Logits 投影 | 原始权重 |
| **Adapter** | RmsNorm + MatMul | Draft → logits | lm_head.weight (共享) |
| **Shadow Projector** | Linear↓ + Linear↑ | KV 缺口填充 | 随机初始化 + 在线蒸馏 |

**编译约束**: 全部编译产物在 `from_loader()` 阶段完成。推理热路径零编译。符合 REQ-JIT-CACHE-004。

### 17.9 自适应调度决策

调度器在每个 `step()` 循环中动态决定是否启用推测解码:

```
if decode_request_count > 0:
    if avg_acceptance_rate > 0.5:        # 历史接受率足够高
        step_speculative()               # 推测解码路径
    else:
        step()                           # 标准解码路径
else:
    step()                               # 纯 prefill / 混合路径
```

**动态回退**: 连续 3 轮 acceptance_rate < 0.3 时，自动回退到标准解码（避免无效 draft overhead）。acceptance_rate 通过 §9.5 Epilogue Telemetry 的 Entropy 指标间接监控。

### 17.10 SAGUARO 多 GPU 并行推测解码 (Speculative Speculative Decoding)

> **学术依据**: SAGUARO (arXiv:2603.03251, ICLR 2026)
> **适用条件**: ≥2 GPU（1× Draft GPU + 1× Target GPU），推荐 5× GPU（1× Draft + 4× Target TP=4）
> **与 §17.1 EESD 的关系**: **正交可组合**。单 GPU 走 EESD (浅层变体做 draft)；多 GPU 走 SAGUARO (独立 draft GPU，draft+verify 并行)。

**核心创新**: 传统推测解码 (SD) 的 draft→verify 是**串行**的。SAGUARO 的 **Speculative Speculative Decoding (SSD)** 将 draft 和 verify 部署在独立 GPU 上并行执行，消除 draft 延迟对 verify 的阻塞。

```
时间线对比:

传统 SD (单 GPU, 串行):
  [Draft K tokens][Verify all K][Draft K][Verify all K]
  ←── 轮 1 ──→←── 轮 2 ──→

SSD / SAGUARO (双 GPU, 流水线):
  Draft GPU:  [D₁ K tokens][D₂ K tokens][D₃ K tokens]
  Target GPU:             [Verify D₁]  [Verify D₂]  [Verify D₃]
                           ↑ 流水线重叠 ↑
```

#### 17.10.1 三个数学最优策略

**Theorem 12 — Geometric Fan-Out (几何扇出)**:
- 传统 SD 每轮验证 1 个 draft 序列。SSD 同时验证 F×(K+1) 个分支
- 最优扇出分配: 位置 k 的分支数 `F_k = F_0 × a_p^(k/(1+r))`
  - `a_p` = token 接受概率
  - `r` = draft/target 速度比
- 设计直觉: 更多计算分配给"大概率短接受"位置 (短 draft 快速拒绝, 长 draft 缓慢接受)

**Theorem 15 — Cache-Aware Sampling (缓存感知采样)**:
- 对已缓存的 top-F token 的概率乘以常数 `C < 1`，将剩余分布质量集中到缓存 token 上
- 当 `C → 0` 时 cache 命中率单调递增
- SAGUARO 选择使 `P(cache_hit) × P(accept | hit)` 最大化的 C 值

**Theorem 17 — Adaptive Fallback (自适应回退)**:
- Cache 未命中时的两种备份:
  - **慢速 (神经网络)**: 用 draft 模型实时生成 — 质量高, 延迟高
  - **快速 (n-gram)**: 随机/n-gram 生成 — 质量低, 延迟低
- 低 batch (`b < b*`) 用慢速备份, 高 batch (`b ≥ b*`) 切换到快速备份
- 阈值 `b*` 由解析公式推导, 平衡延迟和接受率

#### 17.10.2 与 gllm JIT 管线的集成架构

```
┌───────────────────────────────────────────────────────────────────┐
│                    SAGUARO 多 GPU 架构                            │
│                                                                   │
│  ┌─────────────────┐         ┌──────────────────────────────┐    │
│  │  Draft GPU (×1)  │  NCCL   │     Target GPU Pool (×N)     │    │
│  │                  │ ──────▶ │                              │    │
│  │  PolymorphicExec │  every   │  PolymorphicExec             │    │
│  │  L2_hot variant  │  round   │  Full variant                │    │
│  │  + Adapter       │         │  + Postamble                 │    │
│  │                  │         │                              │    │
│  │  SpeculationCache│         │  EqSpec Batch Verify         │    │
│  │  ┌──────────────┐│         │  ┌──────────────────────────┐│    │
│  │  │ draft_logits ││         │  │ tree_attn_mask (F*(K+1)) ││    │
│  │  │ → fan-out    ││         │  │ per-seq KV commit/rollback││    │
│  │  │ → cache_aware││         │  └──────────────────────────┘│    │
│  │  │   sampling   ││         │                              │    │
│  │  └──────────────┘│         │                              │    │
│  └─────────────────┘         └──────────────────────────────┘    │
│                                                                   │
│  流水线时序:                                                       │
│  T=0: Draft GPU 生成 D₁ F*(K+1) 分支                             │
│  T=1: Draft GPU 发送 D₁ → Target, 同时开始生成 D₂                 │
│  T=2: Target 验证 D₁, Draft 同时完成 D₂                           │
│  T=3: Draft 发送 D₂ → Target, Target 返回 D₁ 验证结果             │
│  ...无限循环...                                                    │
└───────────────────────────────────────────────────────────────────┘
```

**NCCL 通信约束**: 每轮仅一次 `all-gather` 同步 (非每 token), 延迟 ~10-100μs/round。Fan-out 策略计算时需纳入此开销。

#### 17.10.3 Speculation Cache — 运行时数据结构

```
SpeculationCache:
  // Key: (prefix_hash, position) → Value: [F 个 token candidates + logits]
  // 大小: O(B × F × K × (K+1) × (V+1)) bits
  // B=batch, F=fan-out, K=draft_length, V=vocab_size
  // 典型: B=8, F=4, K=5, V=128K → ~数百 MB, 可完全容纳于 HBM
  // 刷新策略: 每轮 speculation round 后全量刷新 (旧 cache 对新 prompt 无效)

  // Cache-Aware Sampling (Theorem 15):
  // 标准采样: p(x) = softmax(logits)
  // 缓存感知: p'(x) = p(x) * C   if x in cache_top_F
  //          p'(x) = p(x) * (1 + Σ_unnorm_residual / Σ_total)  otherwise
  // C 值选择: argmax_C [P(cache_hit|C) × P(accept|hit, C)]
```

#### 17.10.4 自适应路径选择: EESD vs SAGUARO

```
Executor::from_loader():
  if gpu_count >= 2:
    mode = SAGUARO                    // 多 GPU: 独立 draft + target
    draft_device = gpu_pool[0]         // 第一张卡做 draft
    target_devices = gpu_pool[1..N]    // 剩余卡做 target (TP)
  else:
    mode = EESD                        // 单 GPU: 浅层变体做 draft
    draft_variant = L2_hot             // 复用自身模型浅层
    target_variant = Full              // 全量模型做验证
```

**关键约束**: 两种模式共享 EqSpec 三不变量 (§17.4)、各向异性树构建 (§17.3) 和硬件级 Compact→Execute→Scatter (§17.6)。区别仅在 draft 来源和并行模式。

---

## §18 交叉机制协调约束 (CROSS-MECHANISM-COORDINATION)

> **L1i 预算协议**: §9.6 定义了代码段分割和变体特化策略，确保所有机制的指令足迹 ≤ 80% L1i。
> **核心原则**: 所有跨机制冲突通过 **编译时变体隔离** 消解，不在运行时引入条件分支。
> **遥测管线**: §13 白嫖点 → §9.5 传输 → §9.2/§9.1/§15.4 消费 → 形成闭环。

### 18.1 遥测信号管线 (Telemetry Signal Pipeline)

所有机制间的协同通信通过 **零额外计算** 的遥测管线完成。不存在独立的"机制间通信通道"——一切寄生在 §13 Epilogue 白嫖网络的尾段指令中。

```
信号产生 (§13, 寄存器内, ~3-10 条 SIMD):
  §13.5  SiLU 死神经元计数 → §13.1 Gate-First Skip
  §13.6  MoE Gate 命中计数 → §15.4 Uncommon Trap
  §13.7  GEMM 行级 ‖row‖₁+max → §13.5 死神经元判定
  §13.8  RmsNorm per-channel scale → §11.2 KIVI K 量化
  §13.9  Softmax 锐度 + Sink 检测 → §11.2 Sink FP16 保护
  §13.10 Embedding ‖embed‖₂ → §11.3 RaBitQ 初始修正
  §13.11 残差方向余弦 cos(θ) → §16.2 Early-Exit 精化指标
  §13.3  跨层能量差 Δρ → §14.3 层跳过决策

信号传输 (§9.5, STG 写入 KV Page Header):
  Epilogue 尾段的 STG 指令 → 写入 KV Page Header padding bytes
  宿主机低频轮询 (后台 Daemon, 不在热路径)
  L1i 影响: 零 — STG 不增加指令足迹

信号消费:
  §9.2  JIT Director Daemon: 冷专家零命中 → Hot JMP 物理封杀
  §9.1  Block Routing: Δρ < ε → Thread Block 直接 Thread Exit
  §15.4 Uncommon Trap: DEOPT_REQUEST → 主机端微冻结恢复
  §17.9 Spec Scheduling: Entropy 低 → 启用推测解码
  §10.2 Adaptive Chunking: Softmax 锐度 → 动态 chunk size
```

### 18.2 Critical 冲突消解清单

#### C1: §9.1 单 Kernel 宪法 vs §12 空间异构分治

**冲突**: §9.1 要求每步仅 launch 1 个 Mega-Kernel。§12 要求不同 SM 分区跑不同图拓扑。

**消解 (ARCH-CROSS-C1)**: 放松为 **"同构子批单 Kernel"**。
- §12.4 Shape Bucketing 将 batch 按 Golden Size 归类
- 同一 Golden Size 的请求打包进同一个 Mega-Kernel launch
- 不同 Golden Size 分批 launch（各自编译独立 Variant）
- **L1i 保证**: 每个 Variant 只包含一种拓扑，无需运行时 `cmp` 判断

```
batch = [64×decode, 128×prefill-chunk, 64×spec-verify]
         ↓ §12.4 归类
launch 1: Variant-Dense-SEQ64   (decode tokens)
launch 2: Variant-Dense-SEQ128  (prefill chunk)
launch 3: Variant-SpecVerify-SEQ64 (spec tree)
顺序 launch，共享 L2 KV cache
```

#### C2: §13.3 残差旁路 vs §9.3 残差总线注入点

**冲突**: 层跳过 (Δρ < ε) 会跳过配置在特定层的 RAG 注入 / Guardrail 探针。

**消解 (ARCH-CROSS-C2)**: **注入点语义锚定 + 最近邻迁移**。
- §16 定义了 `LayerTarget` 语义锚点 (ShallowSyntax/MidSemantic/DeepLogic)
- 残差旁路决策器维护 **Injection Map**: `{layer_idx → [active_injections]}`
- 层 K 被跳过时，K 上的注入点自动迁移到 **最近的未跳过层**
- 迁移约束: 只能迁移到**同语义区**内 (MidSemantic 内部迁移，不跨到 DeepLogic)
- **L1i 保证**: 迁移决策在 Dispatch-Time 完成，Variant 编译时注入点已固化

#### C3: §9.4 MoE 核内分发 vs §15.3 CPU/GPU 并行

**冲突**: CPU 侧专家计算需要 residual DtoH→计算→HtoD，打破单 Mega-Kernel 模型。

**消解 (ARCH-CROSS-C3)**: **AsyncExpertLayer 延迟合并模式**。
- MoE 层在 Mega-Kernel 内仅执行 **GPU 热专家** (Top-1 或 Top-2)
- 冷专家 (低频, 已被 §13.6 判定) 标记为 `DeferredCpuExpert`
- Mega-Kernel 该层结束后，`micro_freeze` (~0.5ms) 等待 CPU async stream 完成冷专家计算
- CPU 结果通过 **shared memory (Metal) 或 HtoD (CUDA/HIP)** 合并进 residual stream
- **L1i 保证**: GPU Variant 不包含 CPU 专家代码 — 那属于 CPU 端的独立 Variant
- **约束**: CPU 专家仅处理 decode token (seq=1)；Prefill chunk 全走 GPU 热专家

#### C4: §9.2 Hot JMP 修补 vs §9.1 Block Routing 并发安全

**冲突**: Mega-Kernel 执行中途 Hot JMP 覆写 .text 导致不同 thread block 执行不同版本。

**消解 (ARCH-CROSS-C4)**: **步骤间隙 + 双缓冲代码页**。
- Hot JMP **严格在 step 间隙**应用（Host 端确认无活跃 kernel launch）
- GPU 端: **双缓冲代码页** — 新代码写入 shadow page → 单次 `cuModuleLoad` 切换
  - 正在运行的 kernel 读旧 page，下次 launch 读新 page
- x86 端: **5-byte atomic JMP** (已原子)，NOP Slide 预留
- **L1i 保证**: 代码页切换是整体替换，不存在半更新状态

#### C5: §9.1 Block Routing vs §9.4 MoE 128+ 专家的 SMEM 爆炸

**冲突**: Compact/Scatter 缓冲区 + 专家路由表 + FFN 权重超出 SMEM 预算。

**消解 (ARCH-CROSS-C5)**: **路由表寄存器化 + 权重仅 L2 预取**。
- §12.7.3 Perfect Hash Jump Table → 编译为 **立即数常量** (不占 SMEM)
- Thread Block 内 `cmp expert_id, #N; je expert_N_code` 链 — 纯寄存器+指令缓存
- 专家权重通过 §15.2 TurboQuant 4-bit 预取，**仅驻留 L2 cache**（不进 SMEM）
- SMEM 仅存: Compact/Scatter index buffer (~1 KB / 128 req) + Attention tile (~4 KB)
- **L1i 保证**: 专家代码按 §9.6 .text.warm 放置，仅热专家常驻 L1i

#### C6: §16.4 Guardrail Safety Veto 的作用域

**冲突**: Safety Veto 是 per-request 的，但 Hot JMP 是全局的（所有 thread block）。

**消解 (ARCH-CROSS-C6)**: **Safety Veto = Block Routing 标记，非代码修改**。
- Guardrail 探针在 §9.6 .text.cold 中执行（不占热路径指令）
- 探针结果写入 `SafetyFlags[request_id]` — 每个 thread block 可读取
- Thread Block 检查自身 request 的 flag → **条件跳过输出写入**（§9.1 Scatter 阶段）
- 这不是 `if (guardrail_active) { ... }` 的条件分支 — 而是 **Scatter index 中排除该 request**
- **L1i 保证**: Guardrail 判定代码 ~50 字节，内联在 Scatter 循环中

### 18.3 High 冲突消解清单

| # | 冲突 | 消解方案 | L1i 策略 |
|---|------|---------|---------|
| H1 | §10 Chunked Prefill + §9.4 MoE 的 SMEM 三重压力 | SMEM 硬分区: Attention 40% + Routing 立即数(0%) + Compact 20% + 余量 40% | 路由表不进 SMEM (C5) |
| H2 | §9.5 Telemetry + §11 KV Quant 争抢 KV Page Header 字节 | Header 扩展为 64B: telemetry 20B + quant_meta 32B + deopt_flags 8B + reserved 4B | 无 L1i 影响 (数据布局) |
| H3 | §10 + §11 Chunked Prefill 中 K per-channel scale 校准不完整 | K scale 延迟策略: prefill 阶段用 per-token scale, prefill 完成后重算 per-channel scale 并固化 | TurboQuant Variant 分 prefill/decode 两版 |
| H4 | §11 4-bit 量化 vs §16 残差总线注入精度损失 | 注入点前后 **局部 FP16 切换**: Injection 点前 2 层用 FP16 residual, 注入后 2 层恢复正常量化 | FP16 路径编译为独立 Variant |
| H5 | §12 + §17 Spec Decode 的 SM 分区预算不足 | 优先级: Prefill > Verify > Draft; Draft 可降级到 CPU 端 (§15.3) | Draft Variant 指令足迹最小化 (~12 KB) |
| H6 | §12.7 constexpr strides vs §10 动态 chunk size | Chunk size 限定为 §12.4 Golden Size 集合 (非任意值) | 每 Golden Size 编译独立 stride 常量 |
| H7 | §17.4 EqSpec KV rollback vs §9.1 Scatter 已写入 | Tree KV 使用**临时 buffer** (L2 cache), 仅 accepted tokens commit 到主 KV | Verify Variant 含 temp buffer scatter |
| H8 | §17.5 Shadow KV + §11 TurboQuant 量化范围不匹配 | Shadow projector 输出 **clamp 到 TurboQuant 量化范围** (calibrated min/max) | Clamp 指令 ~20B 内联 epilogue |
| H9 | §16.4 Guardrail vs §17 Spec Decode 树状 token 回滚 | Guard probe 在 **draft phase** 执行 (浅层变体 residual), veto 时整棵树丢弃 | Draft Variant 包含 micro probe (~200B) |
| H10 | §15.3 CPU 专家 + §10 Chunked Prefill 层同步 | CPU 专家仅处理 decode token (seq=1); Prefill chunk 全走 GPU | CPU/GPU 各有独立 Variant |
| H11 | §9.3 + §17 Draft phase 跳过深层注入点 | Draft phase 注入点**前置到浅层** (L0-L10, §16.3 Intent NLU 区); Verify phase 正常深度注入 | Draft/Verify 各自编译注入点位置 |

### 18.4 变体注册表 (VariantRegistry)

所有消解方案统一通过 **VariantRegistry** 实现。注册表在模型加载时填充，运行时按 key 查找。

```rust
/// 变体注册表 — 所有机制冲突的消解中心
///
/// Key = 场景签名, Value = 预编译的 JIT 产物
/// 查找发生在 build_batch() 阶段 (Dispatch-Time), 不在 Mega-Kernel 内
pub struct VariantRegistry {
    /// key = VariantKey → CompiledVariant
    entries: HashMap<VariantKey, CompiledVariant>,
    /// L1i 预算 (从 DeviceProfile 获取)
    l1i_budget_bytes: usize,
    /// 当前已用 L1i 最大值 (用于编译时检查)
    max_footprint_bytes: usize,
}

/// 变体签名 — 决定需要哪些机制的代码
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantKey {
    /// 模型架构 (决定基础图结构)
    pub arch: ModelArch,
    /// 是否含 MoE 层 (决定专家分发代码)
    pub moe_enabled: bool,
    /// Guardrail 是否激活 (决定 probe 代码)
    pub guardrail_enabled: bool,
    /// 推测解码阶段 (None=标准, Some(Draft/Verify))
    pub spec_phase: Option<SpecPhase>,
    /// RAG 注入是否激活 (决定残差注入代码)
    pub rag_enabled: bool,
    /// 序列长度装筒 (Golden Size)
    pub golden_size: usize,
    /// 量化精度 (决定 TurboQuant FWHT 代码)
    pub quant_type: QuantType,
}

/// 编译产物 — 包含指令足迹统计
pub struct CompiledVariant {
    /// JIT 编译的机器码 (可执行内存页)
    pub code: Vec<u8>,
    /// 指令足迹 (字节), 用于 L1i 预算检查
    pub instruction_footprint_bytes: usize,
    /// 该 Variant 涉及的机制列表 (用于审计)
    pub mechanisms: Vec<MechanismId>,
    /// 所属代码段 (.hot / .warm / .cold)
    pub section: CodeSection,
}
```

**编译时 L1i 预算检查**:

```
fn compile_variant(key: VariantKey) -> Result<CompiledVariant> {
    let graph = build_graph_for_key(&key);
    let code = jit_compile(graph)?;
    let footprint = code.len();

    if footprint > l1i_budget_bytes * 0.8 {
        // 自动降级: 将低优先级机制代码移至 .text.warm
        let demoted = demote_mechanisms_to_warm(&code, priority_order);
        if demoted.footprint > l1i_budget_bytes * 0.8 {
            return Err(L1iBudgetExceeded {
                footprint,
                budget: l1i_budget_bytes,
                suggestion: "减少 batch 并发或禁用部分机制",
            });
        }
    }

    Ok(CompiledVariant { code, instruction_footprint_bytes: footprint, ... })
}
```

### 18.5 禁止的冲突模式 (CROSS-MECHANISM-PROHIBITED)

| 禁止模式 | 原因 | 替代方案 |
|---------|------|---------|
| Mega-Kernel 内运行时 `if moe_enabled` | 条件分支 → L1i miss + 分支预测惩罚 | 编译为独立 Variant, Dispatch-Time 选择 |
| Mega-Kernel 内运行时 `if guardrail_active` | 同上 | SafetyFlags + Scatter index 排除 (C6) |
| Mega-Kernel 内运行时 `if spec_phase == Draft` | 同上 | Draft/Verify 各自编译独立 Variant |
| 热路径包含 Guardrail probe 代码体 | 探针代码 ~2KB, 挤占 L1i | 探针在 .text.cold, 写 flag 触发 (C6) |
| 热路径包含 Shadow KV projector 代码体 | 投影代码 ~4KB, 仅 Early-Exit 用 | 投影器在 .text.cold, micro-freeze 执行 |
| 热路径包含 Late-Fusion RAG 注入代码体 | 注入代码 ~1KB, 仅特定请求用 | NOP Trampoline, 仅激活时覆写为 JMP |
| CPU 专家同步等待在 Mega-Kernel 内 | 破坏单 Kernel 原则, 引入 Spin | micro-freeze 在层间执行, 不在 Kernel 内 |
| Double Compact (§12.4 + §9.1 各压一次) | 浪费 cycle, 丢失位置映射 | §12.4 装筒 = §9.1 Compact, 合二为一 |
| Hot JMP 在 step 执行期间触发 | 代码半更新 → 数据损坏 | 严格 step 间隙 + 双缓冲代码页 (C4) |
| 单个 Variant 包含所有机制代码 | 100+ KB, 远超 L1i | 变体特化, 每场景 ~20 KB |

---

## §19 执行路径集成契约 (EXECUTION-INTEGRATION-CONTRACT)

> **问题根因**: gllm 历史上反复出现优化模块"实现完整、测试通过、但未接入推理热路径"的问题。审计发现模块内部逻辑正确，但决策产出仅被 `log::debug!` 打印后丢弃。根因是 SPEC 只描述了优化模块**做什么**，没有定义**如何接入执行路径**和**接入后如何验证**。
>
> **本节目标**: 从 SPEC 层面建立不可绕过的集成契约，确保每个优化模块的决策产出**物理改变执行行为**，而非仅产生日志。

### 19.1 执行钩子声明 (Execution Hook Declaration)

每个优化模块（§13-§17 中定义）**必须**在 SPEC 中声明以下四项：

| 声明项 | 含义 | 示例 |
|--------|------|------|
| **Hook Point** | 优化在推理管线中的接入位置 | `LayerCallback::pre_node(layer=15)` |
| **Decision Type** | 优化产出的决策类型 | `CallbackAction::SkipThisNode` |
| **Effect Target** | 决策影响的目标组件 | `FusedGraphExecutor.node_loop` |
| **Verification** | 证明决策被执行的验证方法 | "有/无优化时 forward 调用次数不同" |

**禁止的声明**:
- ❌ Hook Point = "Executor::step() 中日志打印" — 这不是执行钩子
- ❌ Effect Target = "日志输出" — 日志不是执行效果
- ❌ Verification = "肉眼检查日志" — 必须自动化验证

### 19.2 Per-Node Callback 架构 (Per-Node Callback Architecture)

> **实现状态**: ✅ IMPLEMENTED — `src/graph/layer_callback.rs` (CallbackAction, LayerContext, LayerCallback trait, CallbackChain), `src/engine/callbacks/` (7 个具体回调), `src/graph/executor.rs::run_with_kv_cache_with_callbacks()`

所有需要在中间层介入的优化模块通过 **Per-Node Callback** 接入 `FusedGraphExecutor` 的节点执行循环：

```
FusedGraphExecutor::run_with_kv_cache_with_callbacks()
  for (node_idx, node) in graph.nodes.iter().enumerate() {
    → callback_chain.pre_node(node_idx, node, hidden_state)    ← 优化模块在此介入
    → match action:
        Continue → execute JIT kernel
        SkipThisNode → skip (gate skip, dead code)
        ExitEarly → break loop (early exit, guardrail veto)
        InjectHidden → replace hidden_state (RAG, knowledge)
    → callback_chain.post_node(node_idx, node, output)         ← 遥测、探针在此介入
  }
```

**零开销保证**: 当 `CallbackChain` 为空时，`run_with_kv_cache()` 走原始路径，无额外分支或函数调用。

### 19.3 集成路径注册表 (Integration Path Registry)

每个优化模块的执行路径集成状态必须在下表中明确记录。**状态只有两种**：

| 状态 | 含义 |
|------|------|
| 🟢 **Integrated** | 模块决策通过 Callback 直接改变 FusedGraphExecutor 执行行为 |
| ❌ **Not Integrated** | 模块存在但决策未接入执行路径（等同于未实现） |

> **铁律**: SPEC 审计时，任何 `❌ Not Integrated` 的模块等同于**未实现**，不接受"逻辑正确但未接入"作为完成状态。

#### 当前集成状态

| 模块 | SPEC 章节 | 回调实现 | Hook Point | Decision Type | 状态 |
|------|----------|----------|------------|---------------|------|
| Early Exit (PGSLE) | §16.2 | `EarlyExitCallback` | `post_node(exit_layers)` | `ExitEarly { logits }` | 🟢 |
| Gate Skip | §13.1 | `GateSkipCallback` | `pre_node(ffn_nodes)` | `SkipThisNode` | 🟢 |
| Late-Fusion RAG | §16.1 | `RagInjectCallback` | `pre_node(fusion_layer)` | `InjectHidden { data }` | 🟢 |
| Guardrail Probe | §16.4 | `GuardProbeRunner` (via GenerationHook) | `post_step()` | `HookDecision::Veto/Terminate` | 🟢 |
| Intent NLU | §16.3 | `IntentRecallCallback` | `post_node(target_layer)` | Recall extraction | 🟢 |
| MoE Dispatch | §15 | `MoeDispatchCallback` | `pre_node(moe_layers)` | Routing setup | 🟢 |
| Speculative Decoding | §17 | `SpecDecodingState` | `step()` draft/verify phase | `draft_budget` per sequence | 🟢 |
| Knowledge Injection | §8.1 | — | `pre_node(target_layer)` | `InjectHidden { data }` | 📋 框架就绪 |

### 19.4 禁止"日志即丢弃"反模式 (LOG-AND-DISCARD PROHIBITION)

> **铁律**: 优化模块的决策产出**禁止仅通过日志输出消费**。日志可以伴随决策，但决策本身必须通过结构化路径（Callback/Event/State）改变执行行为。

**禁止模式**:

```rust
// ❌ LOG-AND-DISCARD: 决策只进日志，不改变执行
let decision = gate_skip_analyzer.analyze(&telemetry);
log::debug!("gate skip decision: {:?}", decision);
// 没有任何代码消费 decision 来跳过 FFN
```

**正确模式**:

```rust
// ✅ CALLBACK: 决策通过 Callback 直接改变执行
// Executor::build_step_callbacks() 中:
if avg_dead_neuron_ratio > 0.5 {
    callbacks.push(Box::new(GateSkipCallback::new(
        num_layers,
        vec![SkipDecision::Skip; num_layers],
    )));
}
// GateSkipCallback::pre_node() 返回 SkipThisNode
// → FusedGraphExecutor 跳过该节点的 JIT kernel 执行
```

**审计命令**:

```bash
# 检测日志即丢弃反模式：决策变量在 log 之后不再被使用
grep -rn "log::debug.*decision" src/engine/ src/jit/ | grep -v callback
grep -rn "log::info.*advice" src/engine/ src/jit/ | grep -v callback
```

### 19.5 新增优化模块集成检查清单

当新增任何优化模块（无论来自哪个 SPEC 章节）时，以下检查清单**必须全部通过**：

| # | 检查项 | 验证方法 | 不通过后果 |
|---|--------|---------|-----------|
| IC-1 | 模块是否声明了 Hook Point？ | SPEC §19.3 表格中有该模块的行 | ❌ 视为未实现 |
| IC-2 | 模块的 Decision Type 是否改变执行？ | 代码中 Callback 返回非 `Continue` 的 action | ❌ 视为未实现 |
| IC-3 | 模块是否有集成测试证明执行行为改变？ | 测试中验证"有/无模块时输出不同" | ❌ 阻塞合并 |
| IC-4 | 模块是否在 `build_step_callbacks()` 中注册？ | 代码中 `callbacks.push(...)` 调用存在 | ❌ 阻塞合并 |
| IC-5 | 禁止日志即丢弃反模式？ | 决策变量在 log 之后仍被消费 | ❌ 阻塞合并 |

### 19.6 Callback 优先级表 (Callback Priority Table)

| 优先级 | 回调 | 触发时机 | SPEC 章节 |
|--------|------|---------|----------|
| 100 | Prefetch | `pre_node` | §13.2 |
| 90 | Knowledge Inject | `pre_node` | §8.1 |
| 80 | RAG Inject | `pre_node` | §16.1 |
| 70 | MoE Dispatch | `pre_node` | §15 |
| 60 | Gate Skip | `pre_node` | §13.1 |
| 50 | Early Exit | `post_node` | §16.2 |
| 40 | Guardrail Probe | `post_step` | §16.4 |
| 30 | Intent Recall | `post_node` | §16.3 |
| 20 | Residual Bypass | `pre_node` | §13.3 |
| 10 | Telemetry | `post_node` | §13/§18 |

> **优先级规则**: 高优先级先执行。`SkipThisNode` / `ExitEarly` 等终止性 action 会跳过后续低优先级回调。

---
