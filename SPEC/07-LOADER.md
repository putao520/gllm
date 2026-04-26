# 模型加载与解析层 (Loader)

> **SSOT 声明**: 本文档是 gllm 模型加载、格式解析、配置推导、架构模板、下载管理的唯一真源。

## 1. 统一加载入口

`Loader::auto(model_id)` 自动探测格式和下载源，返回统一的 `LoadedModel`。

```rust
let model = Loader::auto("Qwen/Qwen3-7B-Instruct")?;
// 自动: 格式探测 → 下载 → 对应解析器 → ModelConfig + Weights + Tokenizer
```

探测优先级：文件扩展名 (`.gguf` / `.onnx` / `.safetensors`) → 文件头 magic bytes → 目录结构。

## 2. SafeTensors 解析器

### 2.1 文件结构

零拷贝 mmap 解析。文件布局：

```
[8 字节: JSON 长度 (u64 LE)] [JSON 元数据] [张量数据 (对齐)]
```

| 步骤 | 实现 |
|------|------|
| Header 解析 | 读取 8 字节长度 → 解析 JSON 元数据 → 张量名称映射到 `{dtype, shape, data_offsets}` |
| 数据访问 | `mmap` 映射文件，张量通过 offset 直接引用，零分配 |
| 对齐 | 张量数据按 8 字节对齐（safetensors 规范） |

### 2.2 支持的数据类型

SafeTensors 文件格式支持的类型代码。运行时精度系统和 Element trait 定义见 `SPEC/04-OPERATORS.md` §1.3 和 §2。

| 类型代码 | Rust 类型 | 字节数 |
|---------|-----------|--------|
| F32 | `f32` | 4 |
| F16 | `f16` | 2 |
| BF16 | `bf16` | 2 |
| I8 / I16 / I32 / I64 | `i8`/`i16`/`i32`/`i64` | 1/2/4/8 |
| U8 / U16 / U32 / U64 | `u8`/`u16`/`u32`/`u64` | 1/2/4/8 |
| BOOL | `u8` (bit-packed) | 1/8 |

### 2.3 扩展元数据

SafeTensors 支持通过 `gllm.quantization` 扩展字段存储量化参数（ARCH-QUANT-METADATA）。

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
    }
  }
}
```

缺失量化元数据时返回明确错误，禁止使用默认值推测。

### 2.4 Load-time Dtype Normalisation & Canonical Layout (ARCH-LOADER-NORMALIZE)

SafeTensors / PyTorch 权重通常以 BF16 / F16 存储，且 `nn.Linear.weight` 的物理布局为 HF 风格 `[out_features, in_features]` row-major。Loader 在上传张量前必须一次性完成两项归一化：

| 归一化 | 输入 | 输出 | 并行化约束 |
|--------|------|------|------------|
| Dtype→F32 | `BF16` / `F16` / `F32` / `F64` 原始字节 | `Vec<f32>` | 必须 rayon 并行；单线程 `chunks_exact().map().collect()` 在大模型（Gemma 4 E2B 9.6 GB BF16，约 4.8 B 元素）上需要 60-120 秒，无法接受 |
| Linear 布局 | `[out, in]` row-major `Vec<f32>` | `[in, out]` row-major `Vec<f32>` + `meta.shape = [in, out]` | 必须使用 cache-blocked (tiled) 转置；naive `out[c * rows + r] = in[r * cols + c]` 写步进为 `rows * 4` 字节（典型 1536 × 12288 ≈ 6 KB），每次 L1 miss，吞吐 50-200 MB/s；tiled 版本每个 `64×64` tile 占 16 KB，读写双命中 L1，rayon 按列-tile 拆分 20 核并行 |

**关键常量**:
- dtype 并行 chunk size: `1024` f32（F32 passthrough）/ `4096` f32（F16/BF16/F64 转换） — 平衡并行 overhead 与 per-worker 局部性
- 转置 tile size: `TILE = 64` f32 — 与 L1 cache line (64 B = 16 f32) 对齐、双维度 tile 在 L1 内可重用

**正确性要求**:
- 并行路径必须与串行 `chunks_exact().map()` 产生 bit-identical 输出（即 `to_bits()` 完全相同）
- Cache-blocked 转置必须与 naive 实现产生 bit-identical 输出（所有 finite f32 值）
- 两者均需附带单元测试证明 bit-exactness（见 `src/loader/mod.rs` test 模块）

**禁止**:
- 单线程 `collect()` dtype 转换（违反 ARCH-LOADER-PARALLEL-CONVERT）
- Naive 双层 `for r × for c` 转置（违反 ARCH-LOADER-CACHE-BLOCKED-TRANSPOSE）
- 改变输出数值（不得以"SIMD fast-path"为借口破坏 bit-exactness）

## 2.5 SafeTensors MXFP4 对处理 (ARCH-MXFP4-SEPARATE)

OpenAI gpt-oss-20b 等模型在 SafeTensors 中将每个 MXFP4 逻辑权重存储为两个物理张量：

| 张量后缀 | 内容 | DType |
|----------|------|-------|
| `<prefix>_blocks` | packed e2m1 nibbles | U8 |
| `<prefix>_scales` | per-block e8m0 scale bytes | U8 |
| `<prefix>_bias` | BF16 bias（不属于 MXFP4 对） | BF16 |

### 2.5.1 规范行为：分离保持，禁止重打包

**铁律**：SafeTensors loader **禁止**将 `_blocks` + `_scales` 重打包为 GGUF 交错格式。

| 规则 | 说明 |
|------|------|
| **禁止 repack_to_gguf_layout** | 分离格式是规范内部格式（ARCH-MXFP4-SEPARATE），不需要也不允许转换为交错格式 |
| **禁止隐藏 `_scales` 为 sidecar** | `_blocks` 和 `_scales` 都作为独立 U8 张量暴露给 `tensor_info()` 和 `iter_tensors()` |
| **禁止 MXFP4 ggml_dtype 覆盖** | 两者都是普通 U8 张量，不经过 `WeightsHandle.quantized` 路径 |
| **保持 `_blocks` 原始字节** | `load_tensor_data("_blocks")` 返回原始 packed nibbles，不做任何转换 |
| **保持 `_scales` 原始字节** | `load_tensor_data("_scales")` 返回原始 scale bytes，不做任何转换 |

### 2.5.2 对检测（仅诊断用途）

`mxfp4_pairing.rs` 的 `scan_mxfp4_pairs()` 仍然检测 `_blocks`/`_scales` 对关系，但仅用于：
- 诊断日志（标记 MXFP4 张量对）
- `mxfp4_scale_map()` 查询（graph template 绑定参考）

检测结果**不改变**任何加载行为。`_blocks` 和 `_scales` 各自作为独立 U8 张量走常规上传路径。

### 2.5.3 数据流

```
SafeTensors 文件:
  <prefix>_blocks  (U8, raw nibbles)
  <prefix>_scales  (U8, raw scales)
       ↓
SafeTensorsLoader (无重打包):
  tensor_info("_blocks")  → Some(U8, shape)
  tensor_info("_scales")  → Some(U8, shape)
  iter_tensors()          → 两者都出现
  load_tensor_data("_blocks")  → raw nibbles bytes
  load_tensor_data("_scales")  → raw scales bytes
       ↓
upload_weights (常规 U8 上传):
  WeightsHandle.tensors["_blocks"]  → raw nibbles bytes
  WeightsHandle.tensors["_scales"]  → raw scales bytes
       ↓
Graph Executor weight_binding:
  weight_ptrs["_blocks"]  → *const u8 (zero-copy)
  weight_ptrs["_scales"]  → *const u8 (zero-copy)
       ↓
JIT lower_moe_dispatch_packed():
  gate_up_blocks_ptr  → 读 packed nibbles
  gate_up_scales_ptr  → 读 scale bytes
       ↓
x86_lower emit_mxfp4_dequant(blocks_ptr, scales_ptr):
  从 scales_ptr[block_idx] 读取 1 字节 scale
  从 blocks_ptr[block_idx * 16] 读取 16 字节 nibbles
  JIT 寄存器内 dequant → 32 f32 值
```

### 2.5.4 GPU 性能考量

分离格式对 GPU 后端至关重要：

| 格式 | Scales stride | Nibbles stride | GPU Coalescing | 有效带宽 |
|------|--------------|----------------|----------------|---------|
| 分离 | 1 (2 的幂) | 16 (2 的幂) | 完美 coalesced | ~100% |
| 交错 | 17 (非 2 的幂) | — | 严重撕裂 | ~7.5% |

GPU MoE dispatch 的瓶颈是 memory bandwidth。交错格式的 stride=17 导致 warp 内每个 thread 的读取地址不对齐，memory transaction 利用率暴跌。分离格式的 stride=1 和 stride=16 都满足 coalescing 要求。

### 2.5.5 GGUF 源的处理

GGUF 文件内部以交错格式存储 MXFP4 数据。GGUF loader 负责在加载时拆分：

```
GGUF file: [scale(1B) | nibbles(16B)] × num_blocks  (stride=17)
    ↓ GGUF loader 拆分
scales[]: [s0, s1, s2, ...]                           (stride=1)
nibbles[]: [n0(16B), n1(16B), n2(16B), ...]           (stride=16)
```

拆分后的数据走与 SafeTensors 相同的后续路径。

## 3. GGUF 零拷贝解析器

### 3.1 文件结构

| 区域 | 内容 |
|------|------|
| Header | magic (`GGUF`), version, tensor_count, metadata_kv_count |
| Metadata KV | 键值对，支持 12 种值类型 + ARRAY 嵌套 |
| Tensor Info | name, n_dims, dims, dtype (GgmlDType), offset |
| Padding | 对齐到 `GGUF_DEFAULT_ALIGNMENT` |
| Tensor Data | 量化 block bytes 原始数据 |

### 3.2 量化类型 (21 种)

GGUF 解析器支持完整 21 种 GgmlDType，通过适配层零拷贝传递给 gllm-kernels。

| 族 | 类型 | 适配层 StorageFormat |
|----|------|---------------------|
| 原生浮点 | F32, F16, BF16 | F32 / F16 / BF16 |
| Classic | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | PackedU8 或 U8 |
| K-Quant | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | PackedU8 按位宽 |
| IQ | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS | PackedU8 按位宽 |
| Ternary | TQ1_0, TQ2_0 | PackedU8 |
| 新格式 | MXFP4 | PackedU8 |

类型映射职责链：GGUF 解析器返回 `GgmlDType` + 原始字节 → 适配层 `map_storage_format()` → `ggml_dtype_to_quant_type()` → JIT 驱动内核选择。

### 3.3 QuantizedTensor 存储

量化张量不经 `upload_weights()` 上传，以原始 block bytes 存入 `WeightsHandle.quantized` HashMap，推理时直接传递给 TurboQuant 定点矩阵乘法内核。

```rust
pub struct QuantizedTensor {
    pub data: Vec<u8>,           // GGUF block bytes，零转换
    pub quant_type: QuantType,   // gllm-kernels 量化类型
    pub shape: Vec<usize>,       // 逻辑形状
    pub ggml_dtype: GgmlDType,  // 原始 GGUF 类型标识
}
```

### 3.4 O1 真实性原则

GGUF 解析器遵守严格真实性约束：

- `architecture()` 从 `general.architecture` 元数据读取，禁止从 Model ID 推断
- `vocab_size` 从 `{arch}.vocab_size` 读取，缺失时返回错误而非使用默认值
- `head_dim` 从张量形状推导，禁止 `unwrap_or(128)` 硬编码
- ARRAY[STRING] 解析正确处理每个字符串的长度前缀

## 4. ONNX 解析器

### 4.1 解析流程

prost protobuf 解析 → Model/Graph/Node/Tensor 结构提取 → Graph Pattern Matching → 融合算子映射。

### 4.2 Graph Pattern Matching

三阶段流程：

1. **Matcher**: 在 ONNX 图中匹配子图模式（如 `MatMul → Add → Softmax → MatMul` = Attention）
2. **Mapper**: 将匹配的子图映射为 FusedKernel 描述（FlashAttention, SwiGLU 等）
3. **Kernel**: 融合描述进入 JIT 编译管线，生成硬件原生代码

不匹配的子图降级为 Atomic 算子（SPEC 授权 Fallback）。

### 4.3 精度元数据

从 ONNX tensor dtype 读取实际精度（F32/F16/INT8），禁止基于文件名推断（O1 真实性原则）。

### 4.4 外部数据加载

ONNX 模型支持外部数据文件（`external_data` 引用）。解析器自动处理 `data_location` 字段，从对应文件加载张量数据。

### 4.5 图优化集成

ONNX 图经过以下优化 pass 后进入 JIT 编译：

| Pass | 功能 |
|------|------|
| PatternFusionPass | 子图模式匹配（FlashAttention, SwiGLU, GQA 等） |
| HardwareFusionPass | 硬件感知融合/降级 |
| ConstantFoldingPass | 常量表达式预计算 |
| DeadCodeEliminationPass | 移除未使用节点 |

## 5. PyTorch 解析器

自研 pickle 解析器支持 HuggingFace Hub 发布的 `.bin` 权重文件。**禁止引入 `tch`/`torch`/`pyo3` 等 PyTorch 运行时依赖**（REQ-ARCH-003），所有解析逻辑完全自研。

### 5.1 文件结构

PyTorch `.bin` 文件本质是 Python `pickle` 序列化的 `OrderedDict[str, torch.Tensor]`。

```
# 文件格式: ZIP archive (PyTorch 使用 zip 格式封装)
archive.zip
  ├── archive/data/0          ← 张量原始字节 (连续存储)
  ├── archive/data/1
  ├── ...
  └── archive/archive.pkl     ← pickle 协议索引 (张量名→偏移映射)
```

### 5.2 Pickle 协议支持

| 协议版本 | opcode 范围 | 支持状态 |
|---------|-----------|---------|
| PROTO 0-2 | 基础 opcode | ✅ 完全支持 |
| PROTO 3 | `SHORT_BINUNICODE`, `BINUNICODE` | ✅ 完全支持 |
| PROTO 4 | `SHORT_BINUNICODE`, `MEMOIZE`, `BYTEARRAY8` | ✅ 完全支持 |
| PROTO 5 | `BYTEARRAY8`, `NEXT_BUFFER` | ✅ 完全支持 |

### 5.3 Tensor 提取流程

| 步骤 | 说明 |
|------|------|
| ZIP 解压 | 解析 ZIP 目录，定位 `archive.pkl` 和 `data/*` 文件 |
| Pickle 解析 | 逐 opcode 解析 pickle 字节流，构建 Python 对象图 |
| Tensor 重建 | 识别 `torch._utils._rebuild_tensor_v2` 调用，提取：storage offset, shape, stride, dtype |
| 数据映射 | 根据 storage 索引定位 `data/N` 文件中的字节切片 |
| DType 映射 | PyTorch dtype → 内部 DType |

### 5.4 DType 映射

| PyTorch dtype | 内部 DType | 字节数/元素 |
|---------------|-----------|------------|
| `torch.float32` | `F32` | 4 |
| `torch.float16` | `F16` | 2 |
| `torch.bfloat16` | `BF16` | 2 |
| `torch.float64` | `F64` → 降精度为 `F32` | 8 → 4 |
| `torch.int64` | `I64` (仅用于 index) | 8 |
| `torch.int32` | `I32` | 4 |
| `torch.uint8` | `U8` | 1 |
| `torch.int8` | `I8` | 1 |
| `torch.bool` | `BOOL` | 1 |

### 5.5 配置文件联合读取

PyTorch `.bin` 权重通常伴随 `config.json`（模型配置）和 `tokenizer.json`（分词器）。Loader 统一入口自动定位并解析：

```
model_dir/
  ├── config.json          ← 模型架构配置（hidden_size, num_layers, ...）
  ├── tokenizer.json       ← 分词器
  ├── model-00001-of-000XX.bin  ← 分片权重
  ├── model-00002-of-000XX.bin
  └── model-000XX-of-000XX.bin
```

分片权重通过 `model.safetensors.index.json`（如果存在）或 `pytorch_model.bin.index.json` 定位张量→文件映射。

### 5.6 与 SafeTensors 的优先级

| 场景 | 优先使用 | 原因 |
|------|---------|------|
| `.safetensors` 和 `.bin` 同时存在 | SafeTensors | 零拷贝 mmap、无 pickle 开销、格式确定性 |
| 仅有 `.bin` 文件 | PyTorch 解析器 | 唯一可用格式 |
| HuggingFace Hub 下载 | 自动选 SafeTensors | Hub 默认优先提供 SafeTensors |

### 5.7 安全约束

- ❌ **禁止 `pickle.loads` / `eval` / 任意代码执行** — pickle 协议必须用安全的 opcode 解释器实现，不反序列化任意 Python 对象
- ❌ **禁止 `torch.load()` 调用** — 不依赖 PyTorch 运行时
- ✅ 白名单 opcode 集合 — 只处理 pickle 中与张量重建相关的 opcode
- ✅ 拒绝未知 opcode → 返回 `LoaderError::UnsupportedPickleOpcode`

## 6. ModelConfig 推导

### 6.1 ModelConfig 核心字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `hidden_size` | usize | 隐藏层维度 |
| `num_hidden_layers` | usize | 模型层数 |
| `num_attention_heads` | usize | Q 头数 |
| `num_key_value_heads` | usize | KV 头数 (GQA) |
| `intermediate_size` | Option<usize> | FFN 中间层维度 |
| `num_experts` | Option<usize> | MoE 专家总数 |
| `vocab_size` | usize | 词汇表大小 |
| `max_position_embeddings` | usize | 最大序列长度 |
| `head_dim` | usize | 单头维度 |
| `rope_theta` | f32 | RoPE 基频 |

### 6.2 Tensor-Driven 配置推导

### 6.1 Tensor-Driven 配置推导

配置优先级（Truth > Hint > Fallback）：

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 1 (最高) | 张量形状 | 从实际权重张量推导 |
| 2 | 模型元数据 | GGUF metadata / ONNX attributes / config.json |
| 3 (最低) | config.json | 仅作为最后回退 |

通用推导规则：

| 字段 | 推导方法 | 张量来源 |
|------|----------|----------|
| `vocab_size` | embedding tensor shape 较大维度 | `token_embd`, `embed_tokens`, `wte` |
| `head_dim` | Q projection tensor output / num_heads | `q_proj`, `attn_q`, `self_attn.q_proj` |
| `hidden_size` | embedding tensor shape | 同 vocab_size 来源 |
| `intermediate_size` | MLP gate/up projection output dim | `mlp.gate_proj`, `ffn_gate` |

### 6.2 格式特定规则

- **GGUF**: 从 `{arch}.embedding_length` 等元数据读取，缺失时从张量形状推导
- **SafeTensors**: 遍历张量列表查找标准命名模式，优先使用张量形状
- **ONNX**: 遍历 `graph.initializer` 查找 embedding/Q projection initializer

### 6.3 合规性

禁止硬编码默认值。如果元数据、张量形状和 `config.json` 都无法提供必需字段，必须返回 `InvalidConfig` 错误。

```rust
// 正确: 从张量推导
let head_dim = q_tensor.shape.max() / num_heads;

// 禁止: 使用硬编码默认值
let head_dim = config.head_dim.unwrap_or(128);
```

## 7. 架构模板

### 7.1 YAML → OnnxGraph → JIT 端到端

YAML 模板定义模型架构，展开为 OnnxGraph，经图优化后 JIT 编译执行。

```
YAML 模板 (src/arch/templates/qwen3.yaml)
  ↓ template.rs 解析
OnnxGraph (统一 DAG 表示)
  ↓ graph/optimizer/ 优化
  │   ├── PatternFusion (FlashAttn, SwiGLU, MoERouting, ...)
  │   ├── HardwareFusionPass (硬件感知降级)
  │   ├── ConstantFolding (常量预计算)
  │   └── DeadCodeElimination (移除未使用节点)
  ↓ JIT 编译 (四阶段管线)
硬件原生机器码
```

核心优势：新模型只需提供 YAML 文件，不需要修改任何 Rust 代码。

### 7.2 架构注册表 (SSOT: YAML 驱动)

`src/arch/registry.rs` 的全局注册表在启动时由 `build.rs` 自动扫描
`src/arch/templates/*.yaml`,把每个 YAML 的 `name` + `extra_aliases`
字段加载为别名映射。**没有任何硬编码的模板名列表 / 架构 token → 模板名
的 match 语句 / 测试断言**。新增架构 = 扔一个 YAML 文件进 `templates/`
目录,无需改 Rust 代码、无需改 SPEC、无需更新注册表测试。

**当前模板清单以目录内容为真源**,SPEC 不再复制一份(那会变成双 SSOT
会陷入不同步)。查看支持模型见 `./SUPPORTED_MODELS.md`。

注册 / 反查契约(由 `registry.rs` 单元测试验证):
- 每个 YAML 的 `name` 自反查 → 自己
- 每个 YAML 的 `{name}ForCausalLM` 自动派生 → 自己
- 每个 YAML 的 `extra_aliases` 列出的别名 → 自己
- 每个 YAML 的 `family` 字段能解析为 `ArchFamily`
- 未在任何 YAML 里出现的 token → `None`

### 7.3 架构解析流程

```rust
// 1. 从元数据读取架构标识
let arch = reader.architecture()?;  // GGUF: "qwen3", config.json: model_type

// 2. 匹配 YAML 模板
let template = registry.resolve(&arch)?;

// 3. 展开为 OnnxGraph（Symbolic Shape）
let graph = template.expand(&model_config)?;

// 4. 图优化 → JIT 编译
let compiled = jit_compile(graph, &device_profile)?;
```

### 7.4 融合权重处理

融合权重（如 QKV 拼接）的分割完全基于 `ModelConfig` 参数驱动：

- Q 维度 = `config.hidden_size`
- KV 维度 = `config.num_key_value_heads * config.head_dim`
- 禁止硬编码任何维度值（如 `3072`, `1024`）

### 7.5 YAML 模板 Schema

每个 YAML 模板必须包含以下顶级字段。模板由 `src/arch/template.rs` 解析为 `ArchTemplate` struct，再由 `expand()` 方法展开为 `OnnxGraph`。

#### 7.5.1 顶级字段

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | string | ✅ | 架构唯一标识（如 `qwen3`, `gemma4`） |
| `family` | string | ✅ | 架构族（`ArchFamily` 枚举值: `Decoder`, `Encoder`, `EncoderDecoder`） |
| `extra_aliases` | string[] | ❌ | 额外匹配别名（如 `["Qwen2ForCausalLM"]`） |
| `description` | string | ❌ | 人类可读描述 |
| `config_bindings` | map | ✅ | 模型配置参数到图模板变量的绑定 |
| `layers` | object | ✅ | 层定义（attention/ffn 结构） |
| `special_ops` | object | ❌ | 非标准算子（MoE, PLE, Vision, Audio 等） |
| `post_layer_ops` | object | ❌ | 层后处理（logit softcap, head routing 等） |

#### 7.5.2 `config_bindings` 规范

将 `ModelConfig` 字段映射为模板内可引用的变量名：

```yaml
config_bindings:
  hidden_size: hidden_size           # ModelConfig.hidden_size → {{hidden_size}}
  num_layers: num_hidden_layers       # → {{num_hidden_layers}}
  num_heads: num_attention_heads      # → {{num_attention_heads}}
  num_kv_heads: num_key_value_heads   # → {{num_key_value_heads}}
  head_dim: head_dim                  # → {{head_dim}}
  intermediate_size: intermediate_size # → {{intermediate_size}}
  vocab_size: vocab_size              # → {{vocab_size}}
  max_seq_len: max_position_embeddings # → {{max_position_embeddings}}
  rope_theta: rope_theta              # → {{rope_theta}}
  rope_partial: rope_partial          # → {{rope_partial}} (默认 1.0)
  eps: rms_norm_eps                   # → {{eps}}
```

**铁律**: 所有模型参数必须通过 `config_bindings` 读取，禁止在模板中出现硬编码数字。`template.rs` 解析时遇到数字字面量必须报错。

#### 7.5.3 `layers` 结构

```yaml
layers:
  # Attention 子图定义
  attention:
    ops:
      - op: FusedQkvRope
        inputs: [hidden_state]
        outputs: [q, k, v]
        params:
          num_heads: "{{num_heads}}"
          num_kv_heads: "{{num_kv_heads}}"
          head_dim: "{{head_dim}}"
          rope_theta: "{{rope_theta}}"
      - op: GQA
        inputs: [q, k, v]
        outputs: [attn_out]
        params:
          num_heads: "{{num_heads}}"
          num_kv_heads: "{{num_kv_heads}}"
      - op: Residual
        inputs: [attn_out, hidden_state]
        outputs: [residual]

  # FFN 子图定义
  ffn:
    ops:
      - op: SwiGLU
        inputs: [residual]
        outputs: [ffn_out]
        params:
          hidden_size: "{{hidden_size}}"
          intermediate_size: "{{intermediate_size}}"
      - op: Residual
        inputs: [ffn_out, residual]
        outputs: [hidden_state]
```

#### 7.5.4 `special_ops` 结构（可选）

用于 MoE、PLE、Vision 等非标准模块：

```yaml
special_ops:
  moe:
    enabled: "{{has_moe}}"           # 条件启用
    ops:
      - op: MoEGate
        inputs: [hidden_state]
        outputs: [router_logits]
        params:
          num_experts: "{{num_experts}}"
      - op: MoEDispatchPacked
        inputs: [hidden_state, router_logits]
        outputs: [expert_output]
        params:
          num_experts: "{{num_experts}}"
          top_k: "{{num_experts_per_tok}}"
  ple:
    enabled: "{{has_per_layer_embed}}"
    ops:
      - op: PerLayerEmbed
        params:
          layer_idx: "{{layer_idx}}"
          dim_per_layer: "{{per_layer_dim}}"
```

#### 7.5.5 模板变量引用规则

| 语法 | 说明 | 示例 |
|------|------|------|
| `"{{var}}"` | 从 `config_bindings` 绑定读取 | `"{{hidden_size}}"` → `3584` |
| `"{{layer_idx}}"` | 运行时注入（层循环索引） | `"{{layer_idx}}"` → `0, 1, 2, ...` |
| `"{{has_moe}}"` | 布尔条件 | `"{{has_moe}}"` → `true`/`false` |

**禁止的引用**:
- ❌ 硬编码维度值（`"3072"`, `"1024"`）
- ❌ 引用不存在的 `config_bindings` 变量（编译时报错）
- ❌ 模板内数学表达式（`"{{hidden_size}} * 4"`），所有计算在 Rust 侧完成

#### 7.5.6 新增架构流程

新增模型架构只需：

1. 创建 `src/arch/templates/<name>.yaml`，填写 §7.5.1-7.5.5 规定的字段
2. 如果架构有特殊算子（非标准 OpKind），先在 `gllm-kernels` 的 `ScalarOpRegistry` 注册 scalar impl
3. 运行 `cargo test`，`build.rs` 自动扫描新模板并注册
4. 无需修改任何 Rust 代码（除非需要新 OpKind）

**验证**: `registry.rs` 单元测试自动验证新模板的 `name` 自反查、`ForCausalLM` 派生、`extra_aliases` 映射。

## 8. 下载源管理

### 8.1 自动回退

HuggingFace 下载失败时自动切换到 ModelScope，无需手动指定来源。

```rust
// 源优先级: HF → MS (自动回退)
let model = Loader::auto("Qwen/Qwen3-7B-Instruct")?;
// HF 失败 → 自动尝试 ModelScope
```

### 8.2 智能源选择与缓存

- **源头隔离**: 缓存目录按 `hf/` + `ms/` 子目录隔离，同源模型避免重复下载
- **校验和验证**: 下载完成后验证文件完整性
- **并行加载**: 多线程并发下载大模型文件的多个分片
- **缓存根目录**: `~/.gllm/models/`，可通过 `GLLM_CACHE_DIR` 覆盖
