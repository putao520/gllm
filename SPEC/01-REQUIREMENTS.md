# gllm 功能需求清单

> **📌 SSOT**: 本文档是 gllm 项目的功能需求唯一真源。

## 1. 模型支持 (REQ-MODEL)

> **SSOT**: 详细列表见 [SUPPORTED_MODELS.md](./SUPPORTED_MODELS.md)
> **Constraint (REQ-MODEL-LATEST)**: 仅支持 **2025年9月** 以后发布的模型。当系列推出新版（如 Qwen3），必须立即废弃旧版（Qwen2.5）。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-MODEL-001** | Embedding 模型支持 | 支持 Next-Gen 嵌入模型 | 覆盖 Qwen3-Embed, BGE-M4 | 🟢 已实现 |
| **REQ-MODEL-002** | Rerank 模型支持 | 支持 Next-Gen 重排序模型 | 覆盖 Qwen3-Rerank, BGE-Rerank-v3 | 🟢 已实现 |
| **REQ-MODEL-003** | Generator 模型支持 | 支持 Next-Gen 稠密/MoE 生成模型 | 覆盖 Qwen3, Llama 4, GLM-4.7/5, Mistral 3 | 🟢 已实现 |
| **REQ-MODEL-004** | MoE 架构支持 | 支持 2026 新一代 MoE | 覆盖 Qwen3-A22B, Llama 4 MoE | 🟢 已实现 |
| **REQ-MODEL-005** | GPT-OSS 模型支持 | 支持 OpenAI GPT-OSS (Fused QKV) 架构 | 1. `registry` 解析正确<br>2. `loader` 正确处理 `c_attn`/`c_proj` 权重<br>3. 能够成功加载 GPT-OSS 权重 | 🟢 已实现 |

## 2. 模型加载与管理 (REQ-LOADER)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-LOADER-001** | HuggingFace 集成 | 原生支持 HF Hub 协议，自动下载模型 | `loader::from_hf("repo/model")` 正常工作 | 🟢 已实现 |
| **REQ-LOADER-002** | 智能缓存系统 | 支持本地缓存管理，避免重复下载 | 1. **源头隔离** (hf/ + ms/ 子目录)<br>2. 同源模型避免重复下载<br>3. 支持校验和验证 | 🟢 已实现 (2026-02-05) [commit: be8b444] |
| **REQ-LOADER-003** | 并发加载 | 支持多线程并发下载和张量加载 | 大模型加载速度显著提升 | 🟢 已实现 |
| **REQ-LOADER-004** | 权重转换 | 支持 safetensors/bin 自动探测与转换 | 兼容主流权重格式 | 🟢 已实现 |
| **REQ-LOADER-011** | GGUF 格式支持 | 支持加载 GGUF 量化模型 (q4_0, q8_0) | 1. 自研零拷贝 GGUF 解析器 (`src/loader/gguf/`)<br>2. 能够加载 GGUF 张量到 Tensor 结构 | 🟢 已实现 (2026-02-05) [commit: bc1a030] |
| **REQ-LOADER-005** | ModelScope 支持 | 支持从魔搭社区(ModelScope)下载模型 | 1. `source` 配置项支持切换源<br>2. 自动处理 ModelScope 特有的文件结构 | 🟢 已实现 |
| **REQ-LOADER-006** | 动态模型发现 | 支持任意 HF Model ID，无需预注册 | 1. `Client::new("org/model-name", ModelKind::Chat)` 自动下载<br>2. 从 `config.json` 自动识别架构<br>3. 无需修改代码即可支持新模型 | 🟢 已实现 |
| **REQ-LOADER-007** | 架构自动识别 | 从模型配置文件自动推断架构类型 | 1. 读取 `config.json` 的 `model_type`/`architectures`<br>2. 匹配到对应的 Adapter<br>3. 支持常见架构的自动探测 | 🟢 已实现 |
| **REQ-LOADER-009** | Registry 清理 | 移除 KnownModel 枚举，实现纯动态加载 | 1. 移除 KnownModel<br>2. 移除硬编码 Repo 信息<br>3. 仅允许动态 Model ID | 🟢 已实现 |
| **REQ-LOADER-010** | Registry 删除与显式用途 | 彻底移除 Registry，API 显式指定 ModelKind | 1. 删除 Registry 与 ManifestOverride<br>2. `Client::new(model_id, kind)` 强制显式传入用途<br>3. 提供 `new_chat`/`new_embedding` 快捷方法<br>4. `manifest_from_config` 不再接受 overrides | 🟢 已实现 |
| **REQ-LOADER-012** | ONNX 格式支持 | 原生支持加载 .onnx 模型文件 (纯 Rust 实现) | 1. 集成官方完整 ONNX Proto 定义 (Enterprise Grade)<br>2. **禁止引入第三方推理引擎** (tract/ort)<br>3. 完整解析 Model/Graph/Node/Tensor 结构<br>4. 支持零拷贝/内存映射加载<br>5. **必须实现 Graph Pattern Matching**<br>6. **必须将子图映射为 Fused Kernels** | 🟢 已实现 (2026-02-05) [commit: 088b9a8] |
| **REQ-LOADER-013** | 自动格式探测 | 自动探测模型文件格式 (safetensors/GGUF/ONNX) | 1. 根据文件扩展名自动识别格式<br>2. 支持从 HF/MS 自动选择对应加载器<br>3. 无需用户手动指定格式 | 🟢 已实现 (2026-02-05) [commit: d16d3ea] |
| **REQ-LOADER-014** | GGUF 量化元数据读取 | 从 GGUF 文件读取量化类型信息 | 1. 从 GGUF 元数据读取 `general.quantization_version`<br>2. 从 GGUF tensor 信息读取实际量化类型 (Q4_0, Q8_0, Q5_K, etc.)<br>3. **禁止基于文件名推断** (Ω1: 真实性原则) | 🟢 已实现 (2026-02-07) [commit: 95c30d9] |
| **REQ-LOADER-015** | ONNX 精度元数据读取 | 从 ONNX 文件读取精度信息 | 1. 从 ONNX tensor dtype 读取实际精度 (F32/F16/INT8)<br>2. **禁止基于文件名推断** (Ω1: 真实性原则) | 🟢 已实现 (2026-02-07) [commit: 95c30d9] |
| **REQ-LOADER-016** | 智能源选择 | HF 不可用时自动切换到 ModelScope | 1. HF 下载失败时自动尝试 ModelScope<br>2. 支持配置优先级 (HF→MS 或 MS→HF)<br>3. 记录源切换日志 | 🟢 已实现 (2026-02-05) [commit: d16d3ea] |
| **REQ-LOADER-017** | 统一加载入口 | 单一 API 支持所有格式和源 | `Loader::auto("repo/model")` 自动探测格式+源 | 🟢 已实现 (2026-02-05) [commit: d16d3ea] |
| **REQ-LOADER-018** | 迻除时模型热切换 | 支持在不重启进程的情况下切换模型 | 1. `client.swap_model(new_model)` API<br>2. 自动释放旧模型显存 (KV Cache & Weights)<br>3. 重新初始化新模型环境<br>4. 线程安全（阻塞新请求直到切换完成） | 🟢 已实现 (2026-02-07) [commit: HEAD] |
| **REQ-LOADER-019** | GGUF 架构元数据读取 | 从 GGUF 文件读取架构信息 | 1. 读取 GGUF 内置 `general.architecture` 字段 (如 "llama", "qwen2", "deepseek")<br>2. **禁止基于 Model ID 推断架构** (Ω1: 真实性原则)<br>3. 如果 GGUF 缺少架构元数据，返回明确的错误而非推测 | 🟢 已实现 (2026-02-07) [commit: 95c30d9] |
| **REQ-LOADER-020** | DeepSeek 架构支持 | 支持 DeepSeek V2/V3/R1 系列 MoE 模型 | 1. 实现 DeepSeekAdapter<br>2. 支持 MoE 架构 (671B 总参数, 37B 激活)<br>3. 从 config.json 识别 `model_type: "deepseek"`<br>4. 支持模型: DeepSeek-V3, DeepSeek-V2-Lite, DeepSeek-R1, **Kimi-K2** (使用 DeepSeek 架构)<br>5. 兼容 SafeTensors/GGUF/ONNX 格式 | 🟢 已实现 (2026-03-15) |
| **REQ-LOADER-021** | 融合权重元数据驱动分割 | 融合权重 (如 QKV) 分割完全基于 config.json，禁止硬编码 | 1. `split_phi4_qkv` 等函数接收 `ModelConfig` 参数<br>2. Q 维度 = `config.hidden_size`<br>3. KV 维度 = `config.num_key_value_heads * config.head_dim`<br>4. **禁止**硬编码任何维度值 (如 3072, 1024)<br>5. 支持同一架构的不同变体 (不同 hidden_size / num_kv_heads)<br>6. **关联**: ARCH-QUANT-METADATA-001, ARCH-LOADER-FUSED-METADATA | 🟢 已实现 (2026-02-07) [commit: HEAD] |
| **REQ-LOADER-022** | 张量驱动配置推导 (Tensor-Driven) | 基于 Tensor Role Matching (Regex) 和张量形状推导配置，优先于硬编码逻辑 | 1. **核心**: 定义 `TensorRole` (Embedding/Attention/FFN)<br>2. **推导**: 纯张量形状推导 `hidden_size`, `num_heads`, `head_dim`<br>3. **禁止**: `if model == "llama"` 硬编码逻辑<br>4. **优先级**: 张量形状 > config.json | 🟢 已实现 (2026-02-08) |


### 2.1 GGUF 量化加载 (REQ-QUANT)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-QUANT-001** | GgmlDType→QuantType 桥接 | 映射 GGUF 量化类型到 gllm-kernels QuantType | `ggml_dtype_to_quant_type()` 覆盖 K-Quant/Classic/IQ 共 21 种类型 | 🟢 已实现 |
| **REQ-QUANT-002** | TensorProvider 量化元数据 | TensorProvider 暴露原始 GGML dtype | `ggml_dtype()` default method，GgufReader 实现返回实际 dtype | 🟢 已实现 |
| **REQ-QUANT-003** | QuantizedTensor 双存储 | WeightsHandle 支持量化/native 双存储 | `quantized` HashMap + `new_with_quantized()`/`quantized_tensor()`/`is_quantized()` | 🟢 已实现 |
| **REQ-QUANT-004** | upload_provider 精度分流 | 量化权重跳过 GPU 重加载 | 量化层级→QuantizedTensor 直接加载，非量化层级→按硬件原生极化 DType 映射上传 | 🟢 已实现 |
| **REQ-QUANT-005** | TensorLookup 量化访问 | TensorLookup trait 支持量化 tensor 查询 | `get_quantized()` default method，WeightsHandle 实现 | 🟢 已实现 |
| **REQ-QUANT-006** | Backend DType 特化 matmul | Backend trait DType 特化矩阵乘法 | JIT 编译 DType 特化向量操作，完全抹除运行时 dispatch | 🟢 已实现 |

## 3. 核心功能 (REQ-CORE)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-CORE-001** | 自动后端检测 | 自动选择 CUDA/CPU (ROCm/Metal 计划中) | 1. `detect_backend()` 检测逻辑完整<br>2. 优先级: CUDA > ROCm > Metal > CPU<br>3. 未实现后端返回 `Unimplemented` | 🟢 已实现 (2026-02-07) [commit: 823e6bd] |
| **REQ-CORE-002** | OOM Halt 硬件截断 | GPU OOM 必须直接引发全进程 Halt，严禁以任何降级/退回 CPU 形式处理 | `OomHaltError` 返回，无降级 | 🟢 架构约束生效 |
| **REQ-CORE-003** | 静态极化量化支持 (TurboQuant) | 支持统一强转极化格式 (INT4/FP8) | 1. 能够加载量化模型权重<br>2. 运行时消除多态混合精度分派，全网强制统一至 TurboQuant 固定块规格<br>3. 废除反量化，只提供定点/微浮点硬派算子 | 🟢 已实现 |
| **REQ-CORE-004** | 精度优先架构 | 系统强制运行在"精度优先"模式 | 1. Mega-Kernel 块级路由确保 Batch 内每个请求 Thread Block 独立计算（§9）<br>2. Chunked Prefill 交织调度确保 Decode 永远零等待（§10）<br>3. **移除** 任何吞吐量优先的妥协配置 | 🟢 已实现 (2026-02-07) [commit: 823e6bd] |

## 4. 高级调度与内存管理 (REQ-SCHED)

> **详细设计**: 见 [SPEC/DOCS/scheduling/hgal-scheduler-algorithm.md](./DOCS/scheduling/hgal-scheduler-algorithm.md)
> **架构原则**: 精度优先。Mega-Kernel 块级路由（§9）确保 Batch 内每个请求的 Thread Block 独立计算，消除跨请求规约误差。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-SCHED-012** | ~~确定性调度 (Deterministic)~~ | ⛔ **已废弃** — 被 §9 Mega-Kernel 块级路由覆盖。Thread Block 独立读取 Request_State_Table，Batch 物理布局不影响计算结果。 | - | ⛔ 已废弃 |
| **REQ-SCHED-013** | ~~阶段隔离 (Phase Isolation)~~ | ⛔ **已废弃** — 被 §10 Chunked Prefill 交织调度覆盖。Prefill Chunk 与 Decode Token 交织塞入同一 Batch，Decode 永远零等待。 | - | ⛔ 已废弃 |
| **REQ-EXEC-001** | ~~串行微批次执行~~ | ⛔ **已废弃** — 与 §9 Mega-Kernel 单次 Launch 全批次并行直接矛盾。串行执行吞吐量损失 10-100x，且块级路由已隔离各请求计算。 | - | ⛔ 已废弃 |
| **REQ-EXEC-002** | ONNX 推理执行引擎 | 实现 ONNX 模型的完整推理执行能力 | 1. **Embedding/Reranker**: 单次前向传播已支持 (BERT/XLM-R/Qwen3)<br>2. **Generator**: 实现生成循环 + KV Cache + Sampling<br>3. 使用 FusedKernel 执行 ONNX 子图<br>4. 支持 ONNX 模型的动态批处理<br>5. 与现有 PagedAttention 调度器集成 | 🟢 已实现 (2026-02-07) [commit: 3a0957d] |
| **REQ-EXEC-003** | ONNX KV Cache 集成 | ONNX Generator 模型的 KV Cache 支持 | 1. 从 ONNX 图中提取 KV 输出张量<br>2. 跨轮缓存 KV 状态<br>3. 支持 PagedAttention 页面分配 | 🟢 已实现 (2026-03-15) |
| **REQ-SCHED-001** | PagedAttention 调度 | 实现自定义的分页注意力调度算法 (HGAL) | 1. 显存碎片率 < 5%<br>2. 支持动态 Block 分配<br>3. **禁止序列内页面分散换出**<br>4. **使用 LIRS 优先级计算** | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-002** | 双缓冲 KV Cache | 支持 GPU 双缓冲调度 (Swap 功能) | 1. Swap-in/Swap-out 已实现<br>2. **Warm-up 保护期机制**<br>3. **页面状态机 (Active/Standby/Swapped/Warm/Protected)** | 🟢 已实现 (2026-02-06) [commit: external-kernels] |
| **REQ-SCHED-003** | 动态批处理 | 支持 Continuous Batching | 1. 吞吐量优于 Static Batching<br>2. **序列完成自动移除**<br>3. **新序列动态加入**<br>4. **BatchAction 决策** (Continue/Complete/Pause)<br>5. **企业级死锁防护** (admit_waiting 无限循环修复) | 🟢 已实现 (2026-02-07) [commit: 823e6bd] |
| **REQ-SCHED-004** | Gang-Aware 调度 | 序列组整体调度，禁止序列内页面分散 | 1. **SequenceGroup 作为换出单位**<br>2. **All-or-nothing within one sequence**<br>3. 优先级调度 (FCFS/Priority) | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-005** | Cache Thrashing 防护 | 防止刚换入的页面立即被换出 | 1. **Warm-up 保护期** (默认 100ms)<br>2. **Thrash 率 < 1%**<br>3. 新换入页面不被选中为受害者 | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-006** | Working Set 检测 | 自动识别高频访问页面并锁定保护 | 1. **自动热页检测** (默认阈值 3 次访问)<br>2. **Protected 状态**<br>3. **保护解除机制** | 🟢 已实现 (2026-02-02) [commit: 063f150] |

| **REQ-SCHED-014** | 自适应 JIT 调度策略 | 引入底层 JIT 决策层，基于实时观测动态调整策略 | 1. **微秒级决策** (<10μs)<br>2. **策略热切换** (Accuracy/Throughput)<br>3. **参数自整定** (动态 Batch/Swap)<br>4. **零运行时开销** (Enum Dispatch) | ✅ 已实现 (2026-02-06) [commit: a7e761b] |
| **REQ-SCHED-015** | 调度器重构基线 | 调度器重构以 `GlobalMemoryManager` 为唯一 KV 管理核心，移除 `vllm2024.rs` 冗余 LMCache 结构 | 1. 删除 `LMCacheConfig/LmcacheState/CacheEntry/CacheHit/CacheLevel` 作为核心路径<br>2. `GlobalMemoryManager` 承担跨请求复用入口<br>3. 架构与 `ARCH-SCHED-REFACTOR-2026` 一致 | 🟢 已实现 (2026-02-11) [commit: 8c41031] |
| **REQ-SCHED-016** | ChunkedConfig 融合页面调度 | 保留 ChunkedConfig，用于 Prefill 分块时间片规划 | 1. 支持 `plan_prefill(prompt_tokens, chunk_size)`<br>2. 仅允许 Prefill 阶段分块，禁止与 Decode 混批<br>3. 与 PagedAttention 页面状态机一致更新 | 🟢 已实现 (2026-02-11) [commit: 8c41031] |
| **REQ-SCHED-017** | 确定性批顺序策略 | 引入 BatchOrderPolicy，默认严格 RequestId 排序 | 1. 默认 `StrictRequestIdOrder`<br>2. 批内请求按 RequestId 严格升序<br>3. 禁止吞吐优先乱序策略作为默认执行路径 | 🟢 已实现 (2026-02-11) [commit: 8c41031] |
| **REQ-SCHED-018** | 双管线会话调度 | 支持 Working/Conversation 双管线隔离调度 | 1. Working 管线可在轮次结束回收<br>2. Conversation 管线可跨轮保留<br>3. 页表、预取、换出策略按管线隔离 | 🟢 已实现 (2026-02-11) [commit: 8c41031] |

> **详细设计**: 见 [SPEC/02-ARCHITECTURE.md §调度器重构架构](./02-ARCHITECTURE.md#调度器重构架构-arch-sched-refactor-2026)

## 4.1 KV Cache 管理重构需求 (REQ-KV)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-KV-001** | KvPrefixIndex 前缀复用 | 无 Session 场景下按最长前缀复用 KV 页面 | 1. `find_longest_prefix(tokens)` 支持 O(n) 前缀匹配<br>2. 支持 append 场景复用（非 hash 全等）<br>3. 命中页面必须校验有效性后复用 | 🟢 已实现 (2026-02-11) [commit: 8c41031] |
| **REQ-KV-002** | SessionKvCache 确定性复用 | 会话内基于 finalized position 做确定性 prefix claim | 1. `register_session`/`claim_session_prefix`/`finalize_session_tokens` API 完整<br>2. `finalized_position` 单调递增<br>3. 禁止 claim 超过已确认边界 | 🟢 已实现 (2026-02-11) [commit: 8c41031] |
| **REQ-KV-003** | KvPipeline 双管线隔离 | Thinking/Reasoning 不污染会话主缓存 | 1. `KvPipeline::{Conversation,Working}` 定义完整<br>2. `prepare_next_turn` 释放 Working，保留 Conversation<br>3. 管线维度参与虚拟页标识 | 🟢 已实现 (2026-02-11) [commit: 8c41031] |


### 4.2 自适应 Chunk 与 KV 增量更新 (REQ-KV-EXT)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-KV-EXT-001** | 自适应 Chunk 大小 | 根据运行时负载（L1 可用页数、并发请求数、prompt 长度）动态调整 prefill chunk_size，替代当前硬编码 max_seq_len | 1. `AdaptiveChunkPolicy` 结构体，输入 L1 可用页数/并发请求数/prompt 长度，输出 chunk_size<br>2. Executor 调用 `plan_prefill()` 时使用自适应 chunk_size 而非 `max_seq_len`<br>3. chunk_size 范围 `[ChunkedConfig::chunk_size, max_seq_len]`，下界为 ChunkedConfig 默认值 64<br>4. 高负载（L1 可用 < 25%）时 chunk_size 缩小至下界<br>5. 低负载（L1 可用 > 75%）时 chunk_size 扩大至 max_seq_len<br>6. 单元测试覆盖高/低/中三种负载场景 | 🟢 已实现 (2026-03-15) |


### 4.3 JIT 缓存协议规范 (REQ-JIT-CACHE)

> **关联规范**: [SPEC/DOCS/scheduling/jit-cache-protocol.md](./DOCS/scheduling/jit-cache-protocol.md)
> **SSOT**: 完整的 REQ-JIT-CACHE-001~006 定义见本文 §8.6。

### 后续增强计划（未来版本）
| 优化 | 说明 | 状态 |
|------|------|------|
| **L3 分布式缓存** | Redis/NATS 等分布式后端支持 | 📋 未来计划 |

## 5. 测试矩阵 (REQ-TEST)

> **测试策略**: 三维测试网格 - 后端 × 模型 × 功能

### 测试维度定义

| 维度 | 选项 | 说明 |
|------|------|------|
| **后端** | `cpu`, `jit-cuda` | `jit-hip`/`jit-metal` 未来支持 |
| **模型类型** | `generator`, `embedding`, `rerank` | 三种核心功能 |
| **模型大小** | `mini` (最小) | 快速回归，CI 友好 |
| **功能模块** | `loader`, `inference`, `scheduler`, `quantization`, `scheduler_refactor` | 分层验证 |

### 测试矩阵需求

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-TEST-001** | 后端覆盖测试 | CPU 和 CUDA 后端功能验证 | 1. 所有单元测试在 CPU 后端通过<br>2. 所有 E2E 测试在 CPU 后端通过<br>3. CUDA 后端可用时通过相同测试 | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-002** | Generator 模型矩阵 | 覆盖所有 Generator 模型类型 | 1. qwen3-7b (基准)<br>2. llama-4-8b (多模态)<br>3. phi-4-mini (轻量)<br>4. qwen3-moe (MoE)<br>5. smollm2-135m (超轻量)<br>6. smollm3-3b (轻量多用途)<br>7. internlm3-8b (高效推理) | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-003** | Embedding 模型矩阵 | 覆盖所有 Embedding 模型 | 1. qwen3-embed<br>2. bge-m3 (中文)<br>3. bge-m4 (升级版)<br>4. e5-small (轻量)<br>5. e5-base (标准)<br>6. e5-large (高精度)<br>7. m3e-base (中文)<br>8. jina-embeddings-v2-small (轻量英文)<br>9. jina-embeddings-v2-base (标准英文)<br>10. jina-embeddings-v4 (最新) | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-004** | Reranker 模型矩阵 | 覆盖所有 Reranker 模型 | 1. qwen3-rerank<br>2. bge-reranker-v2-m3 (轻量中文)<br>3. bge-rerank-v3 (升级版) | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-005** | 功能模块覆盖 | 分层功能测试 | 1. Loader: 权重加载、格式转换<br>2. Inference: 生成、嵌入、重排序<br>3. Scheduler: PagedAttention、CB、Swap<br>4. Quantization: AWQ/GPTQ<br>5. Scheduler Refactor: PrefixIndex、SessionKvCache、KvPipeline、BatchOrderPolicy | 🟢 已实现 (2026-03-15) |
| **REQ-TEST-006** | 量化格式测试 | 多种量化格式验证 | 1. AWQ (已实现)<br>2. GPTQ<br>3. SmoothQuant<br>4. 动态量化 | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-007** | 错误处理测试 | 边界条件和错误场景 | 1. OOM 处理<br>2. 无效输入<br>3. 权重损坏<br>4. 不支持的架构 | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-008** | 性能基准测试 | 吞吐量和延迟验证 | 1. Tokens/sec 吞吐量<br>2. 首token 延迟 (TTFT)<br>3. 内存占用<br>4. 性能回归检测 | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-009** | MoE 专项测试 | MoE 模型特殊验证 | 1. 专家路由正确性<br>2. 负载均衡<br>3. 动态专家选择 | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-010** | 后端一致性测试 | CPU vs CUDA 结果一致性 | 1. 相同输入产生相同输出<br>2. 数值精度在容差范围内 | 🟢 已实现 (2026-02-02) [commit: fc36508] |

### 测试文件规划

| 测试文件 | 覆盖维度 | 状态 |
|----------|---------|------|
| `tests/test_e2e_embedding.rs` | Embedding E2E (REQ-TEST-003) | 🟢 已实现 |
| `tests/test_e2e_generator.rs` | Generator E2E (REQ-TEST-002) | 🟢 已实现 |
| `tests/test_e2e_reranker.rs` | Reranker E2E (REQ-TEST-004) | 🟢 已实现 |
| `tests/test_error_handling.rs` | 错误处理 (REQ-TEST-007) | 🟢 已实现 |
| `tests/test_moe_routing.rs` | MoE 专项 (REQ-TEST-009) | 🟢 已实现 |
| `tests/test_performance.rs` | 性能基准 (REQ-TEST-008) | 🟢 已实现 |
| `tests/quantization_metadata.rs` | 量化格式 (REQ-TEST-006) | 🟢 已实现 |
| `tests/test_backend_compat.rs` | 后端一致性 (REQ-TEST-010) | 🟢 已实现 |

## 6. 架构约束 (REQ-ARCH)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-ARCH-001** | 零拷贝推理 | 推理过程中 GPU 数据不回传 CPU | 符合 ARCH-GPU-001 | 🟢 已实现 |
| **REQ-ARCH-002** | 单一后端原则 | 全程在单一后端执行 | 符合 ARCH-SINGLE-BACKEND | 🟢 已实现 |
| **REQ-ARCH-003** | 纯 Rust 依赖原则 | 禁止引入 `candle`、`tch` 等重量级深度学习框架依赖 | 1. Cargo.toml 中无 candle/tch<br>2. 仅使用 safetensors/half 等底层工具库，GGUF 解析器完全自研<br>3. 计算核心完全自研 (gllm-kernels) | 🟢 已实现 (2026-02-05) [commit: fc36508] |
| **REQ-ARCH-004** | KV Cache Scatter Kernel | GQA 多头 KV cache 写入通过 GPU scatter kernel 一次 launch 完成，禁止逐 token 逐 head 独立 DtoD | 1. 新增 `OpKind::KvScatterWrite` + PTX/HIP/MSL codegen<br>2. grid=(num_kv_heads, seq_len), block=(head_dim)<br>3. MQA 单头保留 2 次 DtoD 快速路径<br>4. GQA 8heads×512seq: 1 次 kernel launch 替代 4096 次 DtoD | 🟢 已实现 (2026-03-22) [commit: b432d26] |
| **REQ-ARCH-005** | GPU 权重常驻缓存 | 首次 forward 一次性上传所有层权重到 GPU，后续 step DtoD 复制 | 1. `GpuWeightCache` 结构体持有 per-layer GPU buffer<br>2. 首次 forward: htod 上传 + 缓存 device_ptr<br>3. 后续 forward: DtoD 从缓存复制到 kernel input<br>4. 大模型 fallback: 可用显存不足时保持 htod 路径<br>5. Executor drop 释放所有 GPU buffer | 🟢 已实现 (2026-03-22) [commit: b432d26] |
| **REQ-ARCH-006** | Metal Prefill KV 直写 | Metal prefill KV write 直接通过 shared memory 指针写入，消除中间 buffer | 1. `metal_write_kv_direct()` 直接 ptr::copy_nonoverlapping<br>2. 消除 dtoh→中间 Vec→htod 三步路径<br>3. 利用 Metal shared memory 特性（`[buffer contents]` CPU 可见） | 🟢 已实现 (2026-03-22) [commit: b432d26] |
| **REQ-ARCH-007** | PagedKvView 三后端统一 | Paged attention 支持 CUDA/HIP/Metal 三后端 | 1. HIP: `gpu_write_kv_cache_scatter_hip` + scatter kernel<br>2. Metal: `metal_write_kv_direct` (shared memory 直写)<br>3. 三后端共享 `build_gpu_paged_attention_graph`<br>4. `GeneratorForwardConfig.paged_kv_page_table` 驱动三后端统一切换 | 🟢 已实现 (2026-03-22) [commit: 6ec3dcd] |

## 观测与调度 (REQ-OBS)

### REQ-OBS-001: Epilogue 物理写入 (In-Place Logging)
- 彻底废除 CPU 侧的 `BasicObserver::capture()` 异步轮询。
- 采用 GPU Kernel 尾部 (Epilogue) 的掩码计算直接通过 `STG` 指令写入 `KvPageHeader`。
- `kv_fragmentation` 和 `logits_entropy` 等指标由 GPU 自主物理内存覆盖，不进入 CPU 主内存通信。
- **验收标准**: 单步 GPU Kernel 执行完毕后，直接抽取 Device 内存对应 PageHeader 取得精确同步特征，零流切换。

### REQ-OBS-002: 单轨 AbsolutePolicy 强制约束
- 剥夺原设计中由 CPU 切换精度模式的权力（移除 AccuracyFirst, Balanced, ThroughputFirst）。
- 系统由 QuantType 直接驱动 JIT 生成硬件原生内核，推理过程中无类型判断分支。
- Scheduler 只能执行护栏级的吞吐保护 `AbsolutePolicy`，只改变 `admit_new_prefill` 等外围流水线压弹，绝不更改核心 `kernel_strategy`。
- **验收标准**: 核心系统中不存在 `kernel_strategy` 传参，任何批次的硬件执行路线拥有绝对的静态等价性。

### REQ-OBS-003: 跨步熔断零介入
- 移除所有基于 CPU 延时反馈的“热切换（set_policy）”。
- Kernel 内部依靠硬编码门控实现条件跳过（Ragged Tensor Compaction）。
- **验收标准**: OOM 或 Veto 直接由硬件指令流阻断（Trap），交由崩溃链或 GC 兜底。

## KV Cache 持久化 (REQ-KV)

### REQ-KV-005: KV Cache 增量持久化
- `update_kv_cache()` 必须将 JIT 计算的 K/V 值写入 cache buffer
- 增量 decode 复用已缓存的 K/V，不重新计算
- **验收标准**: 多步 decode 中 KV cache buffer 包含正确的 K/V 数据

## 错误处理 (REQ-ERR)

### REQ-ERR-001: 消除静默失败
- 所有 `let _ = memory_manager.*` 替换为 `?` 传播或 `log::warn!`
- 所有 `Err(_)` 替换为具体错误匹配
- 所有 `unwrap_or(default)` 替换为 `?` 或显式错误处理
- 所有生产代码 `expect()` 替换为 `Result` 返回
- **验收标准**: `grep -rn "let _ =" src/ | grep -v test` 返回 0 匹配；`grep -rn "Err(_)" src/ | grep -v test` 返回 0 匹配

### REQ-ERR-002: OOM Halt 截断 (ARCH-ZERO-FALLBACK)
- 物理显存分配失败时必须 `log::error!` 记录 `OOM Halt triggered`
- 系统必须当场返回架构级硬件越界错误，坚决禁止退回到 CPU
- **验收标准**: OOM 时直接导致全进程/线程 Halt 并暴露 OOM 异常对象，无任何降级发生

### REQ-ERR-003: Backend Detection 错误传播
- `detection.rs` 的 `expect()` 替换为 `Result` 返回
- 探测失败返回 `Err(BackendContextError)`，不 panic
- **验收标准**: 后端探测失败时返回 Err 而非 panic

## gllm-kernels 侧需求 (REQ-KERNELS)

> **仓库**: `/home/putao/code/rust/gllm-kernels/`
> **关联**: 以下需求在 gllm-kernels 仓库实现，gllm 侧仅记录需求定义和验收标准

### 8.1 IQ Codebook 与 TurboQuant 映射 (REQ-KERNELS-IQ)

> **架构定义**: 所有 IQ 权重的解码均内联静态化至 Mega-Kernel 的整型/定点逻辑中。所有 `dequant` 操作均视为零开销的指令级特征投影，运行时全图保持静态定点状态。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-KERNELS-IQ-001** | IQ1_S 静态映射 | 使用 `IQ1S_GRID`（2048×u64 E8-lattice）实现纯整型查表，禁止升格 f32 | 1. 寄存器内保持量化定点态<br>2. block_bytes=50, QK_K=256 | 🟢 已实现 |
| **REQ-KERNELS-IQ-002** | IQ1_M 静态映射 | 使用 `IQ1S_GRID` 内联定点乘法，禁止运行时解包 f32 浮点 | 1. 寄存器内保持量化定点态<br>2. block_bytes=56, QK_K=256 | 🟢 已实现 |
| **REQ-KERNELS-IQ-003** | IQ2_XXS 静态映射 | 使用 D4-lattice codebook（256×u64） | 1. 寄存器内保持量化定点态<br>2. block_bytes=66, QK_K=256 | 🟢 已实现 |
| **REQ-KERNELS-IQ-004** | IQ2_XS 静态映射 | 使用 `KSIGNS_IQ2XS`/`KMASK_IQ2XS` + D4-lattice | 1. 寄存器内保持量化定点态<br>2. block_bytes=74, QK_K=256 | 🟢 已实现 |
| **REQ-KERNELS-IQ-005** | IQ2_S 静态映射 | 使用 D4-lattice（1024×u64） | 1. 寄存器内保持量化定点态<br>2. block_bytes=82, QK_K=256 | 🟢 已实现 |
| **REQ-KERNELS-IQ-006** | IQ3_XXS 静态映射 | 使用 D4-lattice（256×u32） | 1. 寄存器内保持量化定点态<br>2. block_bytes=98, QK_K=256 | 🟢 已实现 |
| **REQ-KERNELS-IQ-007** | IQ3_S 静态映射 | 使用 D4-lattice（512×u32） | 1. 寄存器内保持量化定点态<br>2. block_bytes=110, QK_K=256 | 🟢 已实现 |
| **REQ-KERNELS-IQ-008** | IQ matmul TurboQuant 集成 | `iq_matmul()` 结合 VNNI/SVE2 指令做直接微字节整数乘加 | 1. 禁止 `dequant→f32` 路径<br>2. 定点产出结果 | 🟢 已实现 |

### 8.2 GPU TileLevelFusion / ComputeRoot (REQ-KERNELS-GPU)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-KERNELS-GPU-001** | GPU TileLevelFusion | `plan_emitter.rs` 支持 `FusionMode::TileLevelFusion`，使用 shared memory 替代 CPU L1 tiling | 1. PTX/HIP/MSL codegen 不再返回 error<br>2. shared memory 分配 = `tile_rows × k × byte_width`<br>3. Norm 输出写入 shared memory，GEMM 从 shared memory 读取<br>4. 与 CPU TileLevelFusion 数值一致（容差 < 1e-5） | 🟢 已实现 (HIP/PTX/MSL 三后端) |
| **REQ-KERNELS-GPU-002** | GPU ComputeRoot | `plan_emitter.rs` 支持 `FusionMode::ComputeRoot`，全量 Norm 输出写入 shared/global memory 后执行 GEMM | 1. PTX/HIP/MSL codegen 不再返回 error<br>2. Norm 全量输出缓冲区分配正确<br>3. 与 CPU ComputeRoot 数值一致（容差 < 1e-5） | 🟢 已实现 (HIP/PTX/MSL 三后端) |
| **REQ-KERNELS-GPU-003** | GPU 融合模式决策复用 | GPU codegen 复用 `detect_tile_vs_compute_root()` 的 75% L1 阈值决策逻辑，GPU 侧用 shared memory 容量替代 L1 | 1. `DeviceProfile` 提供 `shared_memory_per_block()` 方法<br>2. 阈值决策对 GPU 使用 shared memory 容量而非 L1 cache | 🟢 已实现 |

### 8.3 PTX 算法多版本支持 (REQ-KERNELS-PTX-MULTIVER)

> **核心理念**: 同一算法（如 FlashAttention）可维护多个 PTX 实现版本，每个版本针对特定 SM 架构深度优化。运行时通过 JIT 全链路根据 GPU compute capability 动态选择最优内核，**禁止 Fallback**——不支持的 SM 版本必须明确报错。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-KERNELS-PTX-MV-001** | PTX 内核版本注册表 | `PtxKernelRegistry` 支持同一算法注册多个 SM 版本特化实现，每个实现声明 `SmRange { min_sm, max_sm }` | 1. 注册表支持 `register(algorithm, sm_range, emitter_fn)` 注册<br>2. 同一算法可注册多个不重叠的 SM 范围<br>3. SM 范围重叠时编译期/注册期报错 | 🟢 已实现 |
| **REQ-KERNELS-PTX-MV-002** | SM 版本检测与验证 | 启动时通过 CUDA Driver API 检测 GPU compute capability，与注册表匹配 | 1. `CudaDevice` 已有 `sm_version` 字段（现有）<br>2. `PtxKernelRegistry::select()` 返回精确匹配的实现或 `Err`<br>3. 无匹配时返回 `Err("SM {ver} not supported for {algorithm}, requires SM {ranges}")` | 🟢 已实现 |
| **REQ-KERNELS-PTX-MV-003** | 禁止 Fallback 机制 | 不支持的 SM 版本必须返回明确错误，禁止降级到次优实现 | 1. 无 `_ => fallback()` 分支<br>2. 无 `default` 实现<br>3. 错误信息包含当前 SM 版本和所有已注册的 SM 范围 | 🟢 已实现 |
| **REQ-KERNELS-PTX-MV-004** | FlashAttention 多版本实现 | 以 FlashAttention 为首个案例，注册 4 个 SM 版本特化内核 | 1. FA-v1 (sm_70-79): wmma tiled attention<br>2. FA-v2 (sm_80-89): mma.sync + cp.async + Split-Q<br>3. FA-v3 (sm_90-99): TMA + WGMMA + warp specialization<br>4. FA-v4 (sm_100+): TMEM + tcgen05.mma + 2-CTA cooperative<br>5. 每个版本独立 codegen 函数，走完整 JIT 管线 | 🟢 已实现 |
| **REQ-KERNELS-PTX-MV-005** | JIT 全链路集成 | 多版本内核选择集成到现有 JIT 编译管线（Phase 3 ISA Lowering） | 1. `PtxDialect::emit_gemm_kernel` 中 CachedGQA/FlashV2 分支查询注册表<br>2. 选择结果传递到 `kernel_builder` 层<br>3. 生成的 PTX 代码包含正确的 `.target sm_XX` 和对应指令集 | 🟢 已实现 |

### 8.4 深度算子融合 (REQ-FUSION-DEEP)

> **核心理念**: 突破当前 2-5 ops 的融合深度限制，实现 8-16 ops 的深度融合。消除三大瓶颈：单输出 ABI、Opaque Op 黑洞、跨层融合缺失。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-FUSION-DEEP-001** | 多输出 ABI | `CompilerGraph` 支持单个 op 产生多个输出张量，codegen 层支持多输出 kernel | 1. `CompilerOp.outputs: Vec<TensorId>` 已支持多输出（现有）<br>2. JIT ABI 扩展：output 指针数组替代单指针<br>3. MoE pre-attention (Q/K/V) 可融合为单 kernel<br>4. 编译次数从 3× 降为 1× | 🟢 已实现 |
| **REQ-FUSION-DEEP-002** | Attention 可融合化 | 将 MHA/CachedGQA 从 Opaque 降级为可融合的复合 Op，允许前驱/后继融合 | 1. `OpSemantics::Attention` 新分类（非 Opaque）<br>2. RmsNorm → Q/K/V GEMM → RoPE → Attention 可融合为单 kernel (6 ops)<br>3. Attention → O GEMM → Residual Add 可融合为单 kernel (3 ops)<br>4. 融合后消除 4 次中间张量写回 | 🟢 已实现 |
| **REQ-FUSION-DEEP-003** | 深度 Epilogue 链 | EpilogueInjection 支持 ≤8 ops 的 epilogue 链（当前 ≤4） | 1. AVX2: 通过寄存器溢出到栈扩展 epilogue 容量<br>2. AVX-512: 利用 32 个 zmm 寄存器支持 8 ops 无溢出<br>3. GEMM + Bias + SiLU + Mul + Add + RmsNorm 6-op epilogue 可融合<br>4. 融合 cost model 更新：考虑溢出代价 vs 内存流量节省 | 🟢 已实现 |
| **REQ-FUSION-DEEP-004** | FFN 全融合 | SwiGLU FFN 的 Gate/Up/Down 三个 GEMM + 激活融合为单 kernel | 1. Gate GEMM + Up GEMM 共享输入 → QkvSharedInput 模式复用<br>2. SiLU(Gate) × Up → Down GEMM 的 epilogue 链<br>3. 整个 FFN 从 5 个独立 kernel 降为 1-2 个融合 kernel<br>4. 消除 Gate/Up 中间张量写回（2 × hidden × inter × 4 bytes） | 🟢 已实现 |
| **REQ-FUSION-DEEP-005** | 融合规则引擎 | 可扩展的融合规则系统，支持声明式模式匹配 + 硬件感知决策 | 1. `FusionRule` trait: `fn matches(subgraph) -> Option<FusionPlan>`<br>2. 内置规则: NormGemm / GemmEpilogue / QkvShared / AttentionBlock / FFNBlock<br>3. 规则优先级排序（深度融合优先于浅融合）<br>4. 硬件感知: 规则可查询 DeviceProfile 决定融合深度 | 🟢 已实现 |
| **REQ-FUSION-DEEP-006** | 跨层残差融合 | Decoder layer 的残差 Add 与下一层的 RmsNorm 融合 | 1. Layer N 的 `residual_add` + Layer N+1 的 `RmsNorm` 融合为单 kernel<br>2. 消除层间中间张量写回（seq_len × hidden × 4 bytes）<br>3. 需要 `decoder_forward` 层面的图构建支持（跨层边） | 🟢 已实现 |

### 8.5 JIT 图执行通用化 (REQ-JIT-GRAPH)

> **背景**: 当前存在三个相互关联的问题：(1) `CompilerGraph` shape 在构建时硬编码为具体数值，`total_seq` 每步 +1 导致 CachedGQA 图每步重编译；(2) YAML 模板与实际执行路径断层，每新增模型都需在 `decoder_forward.rs` 手写分支。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-JIT-GRAPH-001** | JIT 图 Symbolic Shape 支持 | `CompilerGraph` 支持 symbolic 维度，运行时传入具体值，避免 shape 变化时重编译 | 1. `CompilerGraph` 支持 `SymDim`（符号维度），可声明 `Concrete(usize)` 或 `Symbolic(String)`<br>2. `total_seq`、`batch_size` 等动态维度必须声明为 `SymDim::Symbolic`<br>3. 编译一次，运行时通过 shape binding 传入具体值<br>4. CachedGQA 图不再每 decode step 重编译<br>5. 禁止在图构建时将动态维度硬编码为具体数值 | 🟢 已实现 |

| **REQ-JIT-GRAPH-003** | 图执行器打通（YAML → JIT 端到端） | `OnnxGraph`（从 YAML 展开）直接驱动 JIT 执行，消除 `decoder_forward.rs` 手写分支 | 1. `src/graph/executor.rs` 的 `FusedGraph` 执行器能执行完整 decoder forward（含 KV cache）<br>2. 新模型只需提供 YAML 文件，不需要修改任何 Rust 代码<br>3. YAML → `OnnxGraph` → 图优化 → JIT 编译 → 执行链路端到端跑通<br>4. 现有手写分支（MoE、GQA）逐步迁移到图执行器<br>5. 图执行器支持 symbolic shape binding（依赖 REQ-JIT-GRAPH-001） | 🟢 已实现 |

### 8.6 JIT 编译缓存 (REQ-JIT-CACHE)

> **SSOT**: [SPEC/DOCS/scheduling/jit-cache-protocol.md](./DOCS/scheduling/jit-cache-protocol.md)
> **⚠️ 2026-03-27 架构演进**: 原始 REQ-JIT-CACHE-001~003（基于 `DecodeCachedJit` 函数栈 + `GraphKey{QRope|Norm2, seq_len}` 算子级缓存）已被推翻，由 REQ-JIT-CACHE-001~007 全面替代。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-JIT-CACHE-001** | 全核心态级预编译 | 所有算子必须在模型加载阶段完成全网计算图预编译 | 1. 禁用独立算子的 JIT 缓存支持<br>2. `GraphExecutor` 为唯一的全模级别入口<br>3. 推理热路径中无任何单层或单算子重编译 | 🟢 已设定 |
| **REQ-JIT-CACHE-002** | CPU/GPU 算子与流程统一 | CPU 后端与 GPU 后端的图执行行为必须 100% 对齐 | 1. 消除 CPU 特有算子节点分支<br>2. 共同使用 `OpKind` 泛型拓扑与符号化绑定绑定(`SymDim`)<br>3. 完全消除 `GraphType` 的概念 | 🟢 已设定 |
| **REQ-JIT-CACHE-003** | 磁盘级持久化重构 | 依据版本信息缓存整图的二进制编译形态 | 1. L3 保存全模二进制 Kernel (以 Hash 为签名)<br>2. 跳过应用级调度管线编译<br>3. `Fallback` 防护机制健壮 | 🟢 已设定 |
| **REQ-JIT-CACHE-004** | 热路径零编译 | 推理 Decode Step 层循环中严禁用例 | `compile_graph` 调用频次必须恒为 0 | 🟢 已设定 |
| **REQ-JIT-CACHE-005** | SymDim::Symbolic 动态绑定 | SeqLen 通过 ShapeBinding 运行时绑定 | 1. 结合 JIT 的 变量发射寄存器 (Param Registers) 不触发内核重编译<br>2. 直接于执行队列替换 `total_seq_len` | 🟢 已设定 |
| **REQ-JIT-CACHE-006** | 零层级执行器 | 彻底废除按 Layer、按 Attention/FFN 进行调度的入口 | `ModelJitCache`、`FusedAttentionLayer` 这类概念不得出现任何一行存留 | 🟢 已设定 |

## 9. 终极硬件与通信墙拓扑感知 (REQ-TOPOLOGY)

> **核心理念**: JIT 编译器不仅侦测单一算核，必须将硅晶体系至网络网卡的全部环境常数化，化为 IR 极化限制边界。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-HARDWARE-SENSORS** | 全世代指令核探测 | 涵盖 CPU/GPU 最新前沿硅晶操作集的硬指标识别 | 1. **x86**: `AVX-512` (VNNI, BF16, VP2INTERSECT), `AVX10.1/10.2` (识别 P/E 混合锁定 256-bit Converged), `AMX`, `APX` (31 GPR 全量利用)。<br>2. **ARM**: `SVE2`, `SME/SME2` (`ZA_Array`)。<br>3. **NVIDIA GPU**: Hopper SM90 (`TMA`, `WGMMA`, `cuda::barrier`, `L2 mcast`), Blackwell SM100 (`FP4/FP6 Native Tension`, `Block Scale`)。<br>4. **AMD GPU**: CDNA3 (`WMMA`, XCD/GCD 拓扑屏障) | 🟢 架构约束生效 |
| **REQ-COMM-SENSORS** | 跨域通信墙侦测 | 识别系统级别的异步传输障碍及延迟 | 1. 探明跨 NUMA 核心及缓存 (L1/L2/L3/TLB) 深度，生成 `Core Pinning` 强约束。<br>2. 识别主板 PCIe P2P 与 NVLink/XGMI 跳板限制。<br>3. 探测网卡 `RoCE v2/InfiniBand` 大吞吐 DMA 单向 `RDMA_Latency`。 | 🟢 架构约束生效 |
| **REQ-LOAD-TIME-MATH** | 全格式权重加载 + TurboQuant 运行时优化 | gllm 加载 SafeTensors/GGUF/ONNX 全格式权重，推理过程中执行 TurboQuant 数学精度优化 | 1. 加载任意格式权重文件（解包、内存布局重排、元数据提取）。<br>2. QuantType 直接驱动 JIT 生成硬件原生内核。<br>3. 前向传播中执行 3 个在线 FWHT 旋转（内联 Mega-Kernel Epilogue）。<br>4. KV Cache 非对称量化（K per-channel, V per-token）+ RaBitQ 无偏修正。 | 🟢 架构约束生效 |

## 10. AI 大一统运行时图策略 (REQ-UNIFIED-JIT)

> **极化机制**: "纯 Rust，无软路由退避 (No Fallback)，异构块级调度与热覆写"。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-AI-DEV-GUIDELINES** | Gllm 零妥协禁忌 | 所有开发者/Agent 触犯核心规则直接算架构违章 | 1. 绝对纯 Rust 环境，0 FFI Python 调用。<br>2. 核心发生兼容性不足，必须 `Result::Err` 中断！<br>3. **禁止静默降级到无能的 F32 标量（Silent Scalar Fallback）**。<br>4. 基于纯硬件掩码流出（Ragged Compaction / Predicated bitmask）。 | 🟢 架构约束生效 |
| **REQ-DYNA-BLOCK-GRAPHS** | 基于物理资源的动态专门计算图 | 不采用一图到底，而是根据侦探到的通信/指令墙，派发专属化子图 | 1. JIT 将稠密态、极度稀疏态、残差早期剥离（Early-Exit）分类，为每个特化分支编译不同性能上限的子图。<br>2. 运行期由 `Request_State_Table` 动态下发至 SM 的特定 Thread Block 或指定大小 CPU 核丛。 | 🟢 架构约束生效 |
| **REQ-JIT-HOT-REPAIR** | 运行时原子热修补 (Hot JMP Patching / DCE) | 处理不可逆冷专家或无际长前缀静滞期 | 1. `JIT Director Daemon` 观察大批量的确定稳态池。<br>2. 在无停机、无断点环境下发起 5-bytes 的原子硬件 `.text` 覆盖。<br>3. `jmp`/`NOP` 滑块消除分发指令开销，瞬间重连/坍缩全系统网络图（De-optimization Bailout / Uncommon Traps）。 | 🟢 架构约束生效 |

---

## 11. 全局性能验收水准 (REQ-NEXT-GEN-METRICS)

> **来源**: 会话 6e743114 implementation_plan.md §6
> **核心法则**: 架构设计必须有明确、可量化的高目标索引，而非 "best effort"。

| 硬件级别 | 验收指标 | 说明 |
|----------|----------|------|
| **老卡 (1080Ti / P4 级)** | 通过 TurboQuant 6x KV 压缩 + DTOD 同步削减，**强行**运行原本无法承载的 128K 动态请求，性能无明显崩盘 | 验证极端精度压缩下的可用性与通用性 |
| **当代旗舰 (H100 / AMX)** | 利用 Gate-First 跳过和残差旁路削去 **~50%** 浮点算力需求；利用 TMA 预取消除长 Context 访存延迟 | 极限算力释放 |
| **总吞吐** | 冲击现存开源方案 (vLLM/TGI) 的 **2.5×+** | 同硬件同精度下 End-to-End throughput |
| **Tail Latency (P99)** | < 50ms (混合负载) | Chunked Prefill 交织调度保障 |
| **GPU 利用率方差** | < 15% | 消除 Memory Bound / Compute Bound 切换抖动 |
| **KV Cache 压缩比** | ≥ 4x vs FP16 (即 4-bit 主池) | TurboQuant + QJL 双轨极化 |
| **冷专家 Deopt 恢复延迟** | < 1ms (微冷冻) | Uncommon Trap / OSR Bailout |

---

## 12. Semantic Gatekeeper 知识注入 (REQ-SG)

> **协议 SSOT**: `SPEC/SEMANTIC-GATEKEEPER.md`
>
> **API 定义**: `SPEC/04-API-DESIGN.md §7-§8`
>
> **Callback 集成**: `SPEC/05-OPTIMIZATIONS.md §2.9`
>
> **执行器扩展**: `SPEC/08-EXECUTOR.md §4.2.1`

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-SG-001** | Level Keys 预计算 | 模型加载期通过小 CompilerGraph 预计算 3 个层级键向量（每个检测层） | 1. `LevelKeysCache` 对每个 `detection_depth × num_layers` 填充 3 个非全零 finite 向量；2. 向量维度 = `num_kv_heads × head_dim`；3. 小图复用 `FusedGraphExecutor` 编译 / 运行（ARCH-FULL-JIT 合规）；4. 同一小图在 CPU/PTX/HIP/MSL 产出数值一致（容差 1e-4） | 🔴 待实现 |
| **REQ-SG-002** | FusedAttentionLayer Q-Tap | 检测层 `FusedAttentionLayer` 编译期注入 `q_tap`；JIT codegen 在 `q_proj` 尾段追加 STG 写 ring buffer | 1. `q_tap: None` 时零额外指令（cargo-asm diff 对比）；2. `q_tap: Some(...)` 时主计算结果与无 tap 版本一致（L2 误差 < 1e-4）；3. Ring buffer 读出 Q 与 CPU 参考 `q_proj(hidden)[-1]` 一致；4. 双缓冲 atomic step_index 协议防陈旧读；5. 未拆融合 / 未引入后端特化 GraphType | 🔴 待实现 |
| **REQ-SG-003** | 层级路由与门控 | `SemanticGatekeeperCallback.pre_node` 计算 cosine(Q, K_Lx) 路由；best_score < τ 时返回 Continue | 1. mock Provider 始终返回 None 时 hidden_state 保持不变（cosine 相似度 = 1.0）；2. best_score < gate_threshold 触发 Continue 分支；3. argmax 正确映射到 SemanticLevel::L1/L2/L3 | 🔴 待实现 |
| **REQ-SG-004** | 稳定性追踪与 ActiveState 刷新 | hidden 相似度 > stability_threshold && AST 节点未变时复用 `v_knowledge`；否则 FullCompute；AST 节点变更强制刷新 | 1. 连续 20 步相同语义上下文，ActiveState 复用次数 ≥ 15（≥75%）；2. AST node_kind 变更时 FullCompute 在下一个检测层触发；3. `reset_gatekeeper_state()` 清空 ActiveState 但保留 LevelKeysCache；4. 跨请求边界自动刷新 | 🔴 待实现 |
| **REQ-SG-005** | 残差相加注入（零 API 扩展） | Callback 内部计算 `h_new_last = h_last + α × confidence × v_knowledge`，通过现有 `CallbackAction::InjectHidden` 返回 | 1. 不修改 `CallbackAction` 枚举；2. 不修改 `LayerCallback` trait 签名；3. Provider 返回 confidence=0.0 时 hidden 未变；4. confidence=1.0 + α=0.15 时 hidden 最后 token 数值符合公式（逐元素误差 < 1e-6） | 🔴 待实现 |
| **REQ-SG-006** | KnowledgeProvider + AstSentinel 契约 | 用户实现 trait，SG 内核 trait-object 调度；Provider 失败返回 None 时 SG 降级为 Continue | 1. `KnowledgeProvider::retrieve` 返回 None 时 SG Continue；2. AstSentinel 返回 None 时 SG 仅凭 hidden 锚点做稳定性判断；3. trait 签名严格匹配 `SPEC/04-API-DESIGN.md §7.3-§7.4` | 🔴 待实现 |
| **REQ-SG-007** | 部署形态透明 | SG 对 KnowledgeProvider 的本地 / 远程实现形态无感 | 1. 同一 SG Callback 在本地 LSH Provider 和 HTTP Provider 下行为一致；2. Provider 延迟 0ms 与 500ms 场景下 SG 核心行为（路由、稳定性、注入）等价（仅端到端延迟差异） | 🔴 待实现 |
| **REQ-SG-008** | E2E 行为差异验证 | 注册 SG 后生成的 token 分布与未注册基线有可测量差异，且差异方向与 Provider 返回的知识文本语义一致（NO_ISLAND_MODULE 合规） | 1. 固定 Provider 返回 `"Paris"`；询问 `"Capital of France is"`；对比无 SG 基线，`Paris` token logit 明显提升；2. `SemanticGatekeeperCallback.pre_node` 在真实推理路径（非 `#[cfg(test)]`）被调用 ≥1 次；3. grep `SemanticGatekeeperCallback::pre_node` 在 `src/` 非测试代码中有真实注册点 | 🔴 待实现 |

### 12.1 测试文件规划

| 测试文件 | 覆盖维度 | 状态 |
|----------|----------|------|
| `tests/test_e2e_semantic_gatekeeper.rs` | REQ-SG-001~008 端到端（TEST-SG-001~008 见 `SPEC/SEMANTIC-GATEKEEPER.md §8.2`） | 🔴 待实现 |

### 12.2 实现路径（Destroy-Rebuild 铁律）

- 旧 `src/knowledge.rs` + `src/compat/knowledge_injector.rs` 的 `InjectionKind::FrozenKvChunk / LateFusionVector / DynamicLoRA` 与 `LayerTarget::ShallowSyntax / MidSemantic / DeepLogic` 全部物理删除（CLAUDE.md §6 铁律2 禁止渐进式开发）
- `Client::inject_knowledge(source, target)` 旧签名整体替换为 `Client::register_semantic_gatekeeper(config)`
- 新增 `src/semantic_gatekeeper/` 模块（`level_keys.rs` / `small_graph.rs` / `callback.rs` / `ring_buffer.rs` / `active_state.rs`）
- gllm-kernels 侧 `FusedAttentionLayerConfig` 扩展 `q_tap: Option<QTapConfig>`，Phase 3 codegen 扩展 Q-tap STG 指令生成（x86_64 / AArch64 / PTX / HIP / MSL 全后端）

---

## 13. Head Routing SDK (REQ-HR)

> **协议 SSOT**: `SPEC/HEAD-ROUTING.md`
>
> **API 定义**: `SPEC/04-API-DESIGN.md §3.8`
>
> **定位**: 同一 generator LLM 加载后,通过 Client API 运行时切换输出头形态 (generate / classify_binary / classify_multiway / encode_to_layer),**不重新加载模型权重、不重新 JIT 编译**。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-HR-001** | Binary classify head | lm_head logits 切片 positive/negative token → softmax(T) → P(positive) | 1. `client.classify_binary("Is water wet? Answer yes or no:", {positive="yes", negative="no", T=1.0})` 返回 f32 ∈ [0.0, 1.0]；2. SmolLM2-135M-Instruct 上 P(yes) > 0.5 (真实 LLM 行为)；3. `P(yes) + P(no) - 1.0` 绝对值 < 1e-5 | 🟢 已实现 |
| **REQ-HR-002** | Multiway classify head | N 个候选 token 的 lm_head logit 联合 softmax 归一化 | 1. `classify_multiway(prompt, ["sports", "politics", "technology"])` 返回 Vec<f32> 长度 3；2. `sum(probs) - 1.0` 绝对值 < 1e-5；3. 所有 `probs[i] ∈ [0.0, 1.0]` | 🟢 已实现 |
| **REQ-HR-003** | Mid-layer encode | FusedGraphExecutor 单次前向通过 `MidLayerEncodeCallback` 截断, `encode_to_layer` pool hidden 返回 | 1. `client.encode_to_layer(text, LayerAnchor::Relative(0.5), PoolMode::MeanPool)` 返回 `Ok(Vec<f32>)` 长度 = hidden_size；2. 向量所有元素 finite, L2 norm > 0；3. 不同 anchor (0.5 vs 0.9) 产生**不同** embedding — 证明截断深度真实生效；4. 禁止 stub / scalar fallback / silent Ok | 🟢 已实现 |
| **REQ-HR-004** | 同一 Client 切换不重新加载模型 | 多个 HR API 调用复用同一 `BackendContext` | 1. 同一 client 依次 `generate / classify_binary / classify_multiway`,三次 API 调用之间 `Arc::as_ptr(&state.backend)` 地址恒定；2. HR 调用不触发 `ClientBuilder::build_state`；3. 总 JIT 编译次数 = 1 (加载时) | 🟢 已实现 |
| **REQ-HR-005** | NO_ISLAND_MODULE 合规 | HR API 真实接入生产路径,禁止仅 `#[cfg(test)]` 调用 | 1. `grep classify_binary\|classify_multiway\|encode_to_layer src/*.rs` 在非 `#[cfg(test)]` 路径中有真实实现与导出；2. `Client::classify_binary` 真实调用 `head_routing::LayerAnchor::resolve` / `PoolMode::apply` / `Backend::score_tokens_forward_gpu_pure`；3. E2E 测试 TEST-HR-001/002/004 通过真实 SmolLM2 推理路径验证 | 🟢 已实现 |

### 13.1 测试文件规划

| 测试文件 | 覆盖维度 | 状态 |
|----------|----------|------|
| `tests/test_e2e_head_routing.rs` | REQ-HR-001~005 端到端 (TEST-HR-001..005) | 🟢 已实现 |

### 13.2 实现映射

- `src/head_routing.rs` — `LayerAnchor` / `PoolMode` / `ClassifyBinaryConfig` / `ClassifyMultiwayConfig` / `HeadRoutingError`
- `src/client.rs` — `Client::classify_binary` / `classify_multiway` / `encode_to_layer`
- `src/compat/cpu_backend.rs` — `Backend::score_tokens_forward_gpu_pure` + `encode_at_layer_forward_gpu_pure` + `apply_guardrail_probe` (CPU 完整实现)
- `src/compat/mod.rs` — 相关 trait 方法签名
- `src/compat/gpu_backend_macro.rs` — GPU 后端 (CUDA/HIP/Metal) 统一返回 `Unimplemented` (显式,非 silent)
- `src/graph/executor.rs` — `FusedGraphExecutor::run_with_callbacks` 扩展 (Part 1 基础设施)
- `src/engine/callbacks/mid_layer_encode.rs` — `MidLayerEncodeCallback` (post_node ExitEarly with hidden)
- `src/backend/mod.rs` / `src/engine/executor.rs` — 调用链 pass-through
- `src/lib.rs` — 公共类型导出

---

## 14. Guardrail SDK (REQ-GR)

> **协议 SSOT**: `SPEC/GUARDRAIL.md`
>
> **API 定义**: `SPEC/04-API-DESIGN.md §3.9`
>
> **定位**: 在推理前向**中途**插入小线性安全分类探针 + 多档策略响应 (HaltAndVeto / LogOnly / SampleDowngrade), 正交于 SG / HR。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-GR-001** | 零权重探针不触发 veto | `GuardProbeWeights { weight=[0...], bias=0 }` + `HaltAndVeto { threshold: 0.99 }` | 1. `classify_binary` 成功返回 Ok; 2. `attachment.is_vetoed() == false`; 3. `last_score() ≈ 0.5` (sigmoid(0)); 4. `actual_layer ∈ [0, num_layers)` | 🟢 已实现 |
| **REQ-GR-002** | HaltAndVeto 触发 | `weight=[0...], bias=+20` → sigmoid(20) ≈ 1.0 > threshold=0.5 | 1. `classify_binary` 返回 Err(guardrail vetoed); 2. `is_vetoed() == true`; 3. `last_veto_reason()` 包含 "vetoed" / "score" 子串; 4. `last_score() > 0.99` | 🟢 已实现 |
| **REQ-GR-003** | LogOnly 不改变生成 | `bias=+100, LogOnly` | 1. classify_binary 成功返回; 2. 分数与无 guardrail 时完全相等 (浮点 < 1e-4); 3. `is_vetoed() == false`; 4. `last_score() > 0.99` | 🟢 已实现 |
| **REQ-GR-004** | SampleDowngrade 记录温度 | `SampleDowngrade { min_temperature: 0.3 }` | 1. classify_binary 成功返回; 2. `attachment.downgraded_temperature() == Some(0.3)`; 3. `is_vetoed() == false` | 🟢 已实现 |
| **REQ-GR-005** | Attach / detach 生命周期 | 多次 attach / detach 的 id 管理 | 1. 多次 attach 返回单调递增 id; 2. detach 未知 id → Err; 3. detach 后重复 detach 同一 id → Err; 4. 另一个 id 独立成功 | 🟢 已实现 |

### 14.1 测试文件规划

| 测试文件 | 覆盖维度 | 状态 |
|----------|----------|------|
| `tests/test_e2e_guardrail.rs` | REQ-GR-001..005 + TEST-GR-006 (safetensors 缺文件) | 🟢 已实现 |

### 14.2 实现映射

- `src/guardrail.rs` — `GuardProbe` / `GuardProbeWeights` / `SafetyPolicy` / `GuardrailError` / `GuardrailAttachment` / `GuardrailSharedState` / `load_probe_weights` / `validate_policy` / `resolve_anchor`
- `src/engine/callbacks/guardrail_probe.rs` — `GuardrailProbeCallback` (`LayerCallback::post_node`)
- `src/client.rs` — `Client::attach_guardrail` / `attach_guardrail_inline` / `detach_guardrail` / `build_guardrail_chain` / `guardrails` registry
- `src/compat/cpu_backend.rs` — `Backend::apply_guardrail_probe` CPU 实现
- `src/compat/gpu_backend_macro.rs` — GPU 后端 `Unimplemented` (显式)
- `src/lib.rs` — 公共导出

---

## 15. Intent Recall SDK (REQ-INTENT)

> **协议 SSOT**: `SPEC/INTENT.md`
>
> **API 定义**: `SPEC/04-API-DESIGN.md §3.10`
>
> **定位**: 截断前向到 anchor 层 + pool hidden state 作为意图识别 / RAG query 召回向量。`Client::encode_intent` 是 `encode_to_layer` 的语义包装 (DRY 铁律 — 零代码复制)。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-INTENT-001** | Basic shape + finite | encode_intent 返回 `IntentEncoding { embedding, actual_layer, pool }` | 1. `dim()` == `hidden_size`; 2. 所有元素 finite; 3. `l2_norm() > 0`; 4. `pool` 原样透传; 5. `actual_layer ∈ [0, num_layers)` | 🟢 已实现 |
| **REQ-INTENT-002** | PoolMode 语义 | MeanPool / LastToken / ClsToken 在多 token 文本上产生不同 embedding | 1. 三个 embedding 维度相同; 2. MeanPool vs LastToken L1 delta > 1e-3; 3. ClsToken vs LastToken L1 delta > 1e-3 | 🟢 已实现 |
| **REQ-INTENT-003** | delegate 等价 | `encode_intent` 内部 delegate 到 `encode_to_layer`,结果一致 | 1. 同 text/anchor/pool 下 `encode_intent(...).embedding` 与 `encode_to_layer(...)` 逐元素 \|Δ\| < 1e-5; 2. DRY: `encode_intent` 在 client.rs 中无独立前向实现 | 🟢 已实现 |

### 15.1 测试文件规划

| 测试文件 | 覆盖维度 | 状态 |
|----------|----------|------|
| `tests/test_e2e_intent.rs` | REQ-INTENT-001..003 | 🟢 已实现 |

### 15.2 实现映射

- `src/intent.rs` — `IntentEncoding` / `IntentError`
- `src/client.rs` — `Client::encode_intent` (delegate 到 `encode_to_layer`)
- `src/engine/callbacks/mid_layer_encode.rs` — `MidLayerEncodeCallback` (复用 HR 路径)
- `src/engine/executor.rs` — `Executor::encode_at_layer_for_prompt`
- `src/compat/cpu_backend.rs` — `Backend::encode_at_layer_forward_gpu_pure` CPU 实现
- `src/compat/gpu_backend_macro.rs` — GPU 后端 `Unimplemented` (显式)
- `src/lib.rs` — 公共导出

## 16. CoT Reasoner SDK (REQ-COT)

> **协议 SSOT**: `SPEC/COT-REASONER.md`
>
> **API 定义**: `SPEC/04-API-DESIGN.md §3.11`
>
> **实现模块**: `src/cot_reasoner.rs` + `src/generation.rs::GenerationBuilder::reasoning`
>
> **核心定位**: 对**任意** generator LLM（SmolLM2 / Llama / Qwen 等，不依赖模型自带 thinking_head 权重）原生支持 Chain-of-Thought 推理。完全复用 `Client::generate` 公共管线，不新增 Backend trait 方法。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-COT-001** | Manual 模式 budget 精确控制 | 用户指定 `max_reasoning_tokens` + `step_count`，引擎严格遵守预算与步数上限 | 1. `reasoning_trace.len() ≤ step_count`；2. `total_reasoning_tokens ≤ max_reasoning_tokens`（允许 ≤10% tokenizer 估算误差）；3. `stopped_reason ∈ { StepCountReached, BudgetExhausted }`；4. 最终 `text` 非空 | 🔴 待实现 |
| **REQ-COT-002** | Auto 模式 pattern-match 停止 | `ReasoningMode::Auto` 下任一 `stop_patterns` 子串在 chunk 中命中即停止 | 1. 命中时 `stopped_reason == PatternMatched(p)`；2. 未命中时 `stopped_reason ∈ { BudgetExhausted, EntropyConverged }`，不静默继续；3. `reasoning_trace.len() ≥ 1` | 🔴 待实现 |
| **REQ-COT-003** | Auto 模式 entropy-convergence 停止 | `entropy_threshold: Some(t)` 启用连续 chunk 文本熵估算收敛检测（当前启发式版本，logit 级见 §5.2 未来） | 1. 已知重复模式文本喂入 → 估算 entropy < t → `EntropyConverged`；2. 单元测试覆盖启发式本身；3. 文档说明真实 logit-level entropy 依赖 `GenerationResponse` 扩展 | 🔴 待实现 |
| **REQ-COT-004** | Reasoning trace 完整保留 | 每个 reasoning step 产出的 chunk text 作为独立元素存入 `reasoning_trace: Vec<String>` | 1. `reasoning_trace.len() == actual_steps`；2. 每个 trace 元素非空；3. trace 中保留 step chunk 的原始语义内容 | 🔴 待实现 |
| **REQ-COT-005** | 与 Semantic Gatekeeper 正交 | `register_semantic_gatekeeper(cfg)` 后调用 `reason(...)` 可正常工作，step 间的 `Client::generate` 自然走 SG 注入 | 1. `reason` 调用不 panic；2. 每次内部 `execute_generation` 触发 SG callback ≥1 次（mock counter 验证）；3. 不修改 SG API 或 Callback chain | 🔴 待实现 |
| **REQ-COT-006** | NO_ISLAND_MODULE 合规 | `Client::reason` 必须在真实 SDK 路径被调用，不是孤岛模块 | 1. `GenerationBuilder::reasoning` → `ReasoningBuilder::execute` → `Client::reason` 的转发链真实存在；2. E2E 测试 `test_cot_006_arbitrary_llm` 用 SmolLM2-135M-Instruct 跑通全链；3. grep `Client::reason` 在 `src/` 非测试代码中有至少 1 个真实调用点 | 🔴 待实现 |
| **REQ-COT-007** | Step Hook trait 定义与生命周期 | `ReasoningStepHook` trait 提供 `on_step_start` / `on_step_end` 回调，通过 `ReasoningBuilder::with_step_hook()` 注册 | 1. trait 有 `on_step_start(&mut self, &StepContext) -> StepAction` + `on_step_end(&mut self, &StepResult) -> StepKnowledge` 两个方法；2. trait bound 包含 `Send + Sync`；3. E2E 测试: 注册 hook 后 Manual 3 步 reasoning，`on_step_start` 和 `on_step_end` 各被调用 3 次；4. 不注册 hook 时行为与原实现完全一致 (backward compatible) | 🔴 待实现 |
| **REQ-COT-008** | StepContext 包含 accumulated reasoning text | `StepContext` 传递前序步骤累积文本，允许 hook 基于历史做动态决策 | 1. `on_step_start` 的 `StepContext.accumulated_text` 在 step_index=0 时为空串；2. step_index=1 时包含 step 0 的 chunk_text；3. step_index=2 时包含 step 0+1 的累积文本；4. `model_name` 非空；5. `remaining_budget` > 0 且逐步递减 | 🔴 待实现 |
| **REQ-COT-009** | StepAction 支持 Continue/Skip/InjectPrompt/Halt | `StepAction` 枚举提供 4 种流程控制，`StepKnowledge` 支持跨步知识注入与温度覆盖 | 1. `Continue` 正常执行；2. `Skip` 跳过某步且 trace 无空元素；3. `InjectPrompt(extra)` 的 extra 出现在该步 prompt 中；4. `Halt(reason)` 提前终止且 `stopped_reason = HaltByHook(reason)`；5. E2E 测试覆盖 `InjectPrompt`: 注入关键词后 chunk 包含该关键词 | 🔴 待实现 |

### 16.1 测试文件规划

| 测试文件 | 覆盖维度 | 状态 |
|----------|----------|------|
| `tests/test_e2e_cot_reasoner.rs` | REQ-COT-001~009 端到端（SmolLM2-135M-Instruct 真实模型） | 🔴 待实现 |
| `src/cot_reasoner.rs` `#[cfg(test)]` | 模板渲染 / budget 分配 / stop pattern / entropy 启发式 / Step Hook 单元测试 | 🔴 待实现 |

### 16.2 实现路径（纯 Client SDK，零 Backend 扩展）

- `Client::reason(prompt, mode, template)` 在 `src/cot_reasoner.rs` 实现，内部多轮调用 `Client::execute_generation` 复用现有 JIT 缓存与 FusedGraphExecutor
- `GenerationBuilder::reasoning(mode) -> ReasoningBuilder` 在 `src/generation.rs` 添加便捷链式 API
- **禁止**修改 `src/compat/` 任何 forward pass / Backend trait
- **禁止**新增 `FusedGraphExecutor` 方法
- **禁止**为 CoT 扩展 JIT 管线或 GraphType
- 完全符合 CLAUDE.md ARCH-CPU-GPU-UNIFIED（CoT 只在 Client 层 orchestrate，后端零变更）

## 17. 多模型融合管线 (REQ-PIPELINE)

> **API 定义**: `SPEC/04-API-DESIGN.md §2.1 (Client Builder)`, `§3.5 (Embed+Rerank)`, `§3.6 (Embed+Rerank+LLM)`
>
> **E2E 测试**: `tests/test_e2e_fusion.rs` (REQ-PIPELINE-004/005 融合管线), `tests/test_e2e_rag_pipeline.rs` (REQ-PIPELINE-005 跨模型 RAG)
>
> **核心定位**: 在同一 Client 实例内组合 Embedder + Reranker + Generator 三类模型，支持从单一 embed 到完整 RAG 的渐进式管线，模型间零权重重载、encoder 架构相同时权重共享。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-PIPELINE-001** | Client Builder 多模型挂载 | `Client::builder().model(emb).reranker(rr).generator(llm).build()` 在单次 build 中加载 1-3 个异构模型，各自独立 JIT 编译 | 1. `.model()` 必选；`.reranker()` / `.generator()` 可选；2. 三个模型架构可完全不同（Encoder/Decoder/Reranker）；3. build 失败时所有已加载模型资源释放；4. `model_info()` 返回三者各自的 manifest | ✅ |
| **REQ-PIPELINE-002** | 异构模型 Session 隔离 | Embedder / Reranker / Generator 各自持有独立的 `ClientState`（权重、JIT cache、KV cache），不共享指针或缓存 | 1. 三个 `model_info().id` 互不相等；2. 权重内存地址互不重叠；3. JIT cache 按 `ModelArchKey` 隔离；4. 一个模型 swap 不影响其余两个 | ✅ |
| **REQ-PIPELINE-003** | Encoder 权重共享 | 当 reranker 与 embedder 同架构（相同 `ModelArchKey`）时，encoder 前向权重物理共享，仅输出层独立 | 1. 同架构 embed+rerank 内存占用 < 2× 单 embedder + 10%；2. 两者 embedding forward 数值 bit-exact 一致；3. 不同架构时各自独立加载，无共享 | ✅ |
| **REQ-PIPELINE-004** | Embed+Rerank 融合管线 | `embed_builder(texts).rerank_query(q).top_n(n).generate()` 在单次调用内完成 embed → rerank → 排序截断 | 1. 返回 `EmbeddingsResponse { embeddings, rerank_scores }`；2. rerank_scores 非退化、降序排列；3. top_n 截断后结果数 ≤ n；4. 未设 `.rerank_query()` 时退化为普通 embed（零开销）；5. 设 `.rerank_query()` 但未挂载 reranker → `RerankerNotLoaded` 错误 | ✅ |
| **REQ-PIPELINE-005** | Embed+Rerank+LLM 完整 RAG 管线 | `embed_builder(texts).rerank_query(q).top_n(n).generate_answer(prompt)` 三阶段串联：embed → rerank → LLM 生成 | 1. 返回 `RagResponse { text, sources, rerank_scores }`；2. `text` 非空且语义合理（非重复/退化）；3. `sources` 为 top-n 文档索引集合；4. 未挂载 generator 时 → `GeneratorNotLoaded` 错误；5. LLM context 通过文本拼接传入（非 hidden state 注入），保持通用性 | ✅ |

### 17.1 测试文件规划

| 测试文件 | 覆盖维度 | 状态 |
|----------|----------|------|
| `tests/test_e2e_fusion.rs` | REQ-PIPELINE-001/004/005 融合管线（同架构+跨架构 embed+rerank、top_n 截断、RAG 生成） | ✅ |
| `tests/test_e2e_rag_pipeline.rs` | REQ-PIPELINE-001/002/005 跨模型 RAG（独立 Client session 隔离、cosine 召回+rerank 精排+generator 生成） | ✅ |

## 18. Mega-Kernel Session & Multimodal 支持 (REQ-MEGA)

> **SSOT**: [SPEC/GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md §1.5.3, §6.5, §6.6]
> **背景**: `generate_with_session` 和 `generate_with_multimodal` 已删除 step-by-step Rust 编排路径（`9fa0f51`），改为返回 Err。需要 mega-kernel 原生支持才能恢复。

### 18.1 Session KV Cache 复用

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-MEGA-SESSION-001** | Mega-Kernel Session 模式 | mega-kernel 支持 KV cache 跨轮次复用 | 1. `MegaKernelBusinessConfig.session_enabled=true` 时图中插入 `SessionKvRestore` op<br>2. ABI 新增 `session_position` 参数 (StackArg(104))<br>3. session_position > 0 时跳过已处理 tokens，复用已有 KV cache<br>4. session_position == 0 时 NOP (全新生成)<br>5. `generate_with_session()` 不再返回 Err，改为调用 mega-kernel<br>6. 多轮对话中 KV cache 连续性正确（第二轮输出与第一轮衔接） | 🔴 待实现 |
| **REQ-MEGA-SESSION-002** | Session 位置指针管理 | Executor 层维护 session position 跨轮次传递 | 1. `MegaKernelExecutor` 新增 `session_position: usize` 状态<br>2. 每次 mega-kernel 调用后更新: session_position += generated_tokens<br>3. session reset (新对话) 时 session_position = 0<br>4. 位置溢出保护 (session_position + prompt_len < max_seq_len) | 🔴 待实现 |

### 18.2 Multimodal Fused Hidden 注入

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-MEGA-MM-001** | Mega-Kernel Multimodal 模式 | mega-kernel 支持预计算 fused hidden state 注入 | 1. `MegaKernelBusinessConfig.multimodal_enabled=true` 时图中插入 `MmHiddenInject` op<br>2. ABI 新增 `fused_hidden_ptr` (StackArg(112)) + `num_mm_tokens` (StackArg(120))<br>3. fused_hidden_ptr != NULL 时循环 ADD 到 embedding buffer<br>4. fused_hidden_ptr == NULL 时 NOP (纯文本)<br>5. `generate_with_multimodal()` 不再返回 Err，改为调用 mega-kernel<br>6. 多模态输入的生成结果语义正确 | 🔴 待实现 |
| **REQ-MEGA-MM-002** | Fused Hidden 预计算 | Client 层在 mega-kernel 调用前预计算多模态 fused hidden | 1. `Client::generate_with_multimodal()` 内部先调用 vision/audio encoder 获取 hidden state<br>2. 预计算结果通过 fused_hidden_ptr 传入 mega-kernel<br>3. 编码过程不触发 JIT 重编译（复用已有 encoder mega-kernel） | 🔴 待实现 |

## 19. 自动指令选择器 (REQ-AIS)

> **SSOT**: [SPEC/01-JIT-PIPELINE.md §5.1], CLAUDE.md ARCH-AUTO-INSTR-SELECT
> **背景**: TraceOp→VmInstr 和 OpKind→lower 的手写 match arms 是 bug 的系统性源头

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-AIS-001** | TraceOp→VmInstr 自动查表 | TraceOp body 编译为 VmInstr 序列无需手写 match | 1. `auto_lower_trace()` 覆盖全部 17 个已实现 TraceOp<br>2. 同类操作共享辅助函数（6 二元/5 一元/3 超越函数）<br>3. 未实现 TraceOp 返回 Err（NO_SILENT_FALLBACK）<br>4. 生成结果与原手写 `lower_trace_body` 数值 bit-exact | 🟡 Phase 1 已实现 |
| **REQ-AIS-002** | OpKind→ComputePattern 自动分发 | `emit_standalone_op` 从 29 个手写 match arm 改为 ComputePattern 驱动分发 | 1. Elementwise ops 全部走 auto_dispatch_elementwise 路径<br>2. Norm/Gemm/Attention 等走专用 lower 函数<br>3. MoERouter 有专用 lower（修复 GPT-OSS "CapCapCap"）<br>4. 未实现 OpKind 返回 Err（不静默 NOP）<br>5. 所有 E2E 测试通过（SmolLM2, GPT-OSS-20B） | ✅ 已实现 [commit: gllm-kernels e8ee8460] |
| **REQ-AIS-003** | TraceOp 扩展（Compare/Cast） | 新增 Compare/Cast TraceOp 解锁条件分支和 dtype 转换 | 1. TraceOp::Compare → VmInstr::VecCmp<br>2. TraceOp::Cast → VmInstr::VecCast<br>3. 对应 VmInstr 在 x86_64 和 AArch64 codegen 中实现<br>4. 单元测试验证数值正确性 | ✅ 已实现 [auto_select.rs emit_cmp/VecCast] |
| **REQ-AIS-004** | TraceOp 扩展（HReduce） | 新增 HReduce TraceOp 解锁 softmax/norm 全自动 lowering | 1. TraceOp::HReduce → VmInstr::VecReduce<br>2. 支持 Sum/Max/Min 归约操作<br>3. Softmax 可完全通过 SymExec trace 自动 lowering<br>4. E2E 测试通过 | ✅ 已实现 [auto_select.rs HReduce Sum/Max/Min/Prod] |

## 20. MoE 算子完善 (REQ-MOE)

> **SSOT**: [SPEC/04-OPERATORS.md §4.6]
> **背景**: MoERouter 无 JIT lowering，是 GPT-OSS-20B "CapCapCap" 根因

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-MOE-001** | MoERouter JIT Lowering | 实现 `lower_moe_router()` 分解为 GEMM + softmax + top-k | 1. `emit_standalone_op` 有 `OpKind::MoERouter` match arm<br>2. 内部调用 `emit_gemm_inline_with_hook()` + `lower_reduction_softmax()` + top-k<br>3. 输出 (router_weights, router_indices) 格式正确<br>4. GPT-OSS-20B E2E 测试输出非退化（不再是 "CapCapCap"） | 🔴 待实现 |
| **REQ-MOE-002** | MoEDispatchPacked JIT Lowering | 实现专家分发+计算的融合 lowering | 1. `emit_standalone_op` 有 `OpKind::MoEDispatchPacked` match arm<br>2. 按 router_indices 分发 hidden 到对应专家 FFN<br>3. 加权求和输出<br>4. DeepSeek-V3 / GLM-5 MoE E2E 测试通过 | 🔴 待实现 |

## 21. 统一图来源：YAML 模板驱动 mega-kernel (REQ-UGS)

> **SSOT**: [gllm-kernels SPEC/GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md §2.4-2.6]
> **背景**: `graph_builders.rs` 中的手写图构建函数与 YAML 模板不同步，导致 Gemma-4 E2B 的 8 个系统性 BUG（缺失 3 个 norm + layer_scalar + RoPE cache theta 错误 + 权重布局不匹配）

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-UGS-001** | OnnxGraph→CompilerGraph 转换器 | 新增转换器将 YAML 模板展开的 OnnxGraph 转为 CompilerGraph | 1. `OnnxGraphConverter::convert()` 实现完整的 op_type→OpKind 映射<br>2. 所有已有 YAML 模板（Llama/Qwen3/Gemma4/Phi4/Mistral/DeepSeek）转换结果正确<br>3. 转换器处理条件节点（only_if）、repeat 展开、残差连接 | 🔴 待实现 |
| **REQ-UGS-002** | 异构层模板折叠 | 转换器自动检测重复层模式，折叠为 K 个模板 + hetero_loop | 1. Gemma-4 E2B 35 层折叠为 4 模板 (sliding_small/full_small/sliding_large/full_large)<br>2. 权重布局自动推导（14 个权重 vs 11 个）<br>3. PerLayerWeightLayout 从 YAML nodes 自动计算，替代手写 `compute_per_layer_bytes`<br>4. HeteroLayerLoopConfig 从 config.json layer_types 自动生成 | 🔴 待实现 |
| **REQ-UGS-003** | 删除 graph_builders.rs 手写图构建 | 废弃 `decoder_model()`, `decoder_model_hetero()`, `build_layer_body()` | 1. `graph_builders.rs` 中以上函数删除<br>2. mega-kernel 编译改为消费 OnnxGraphConverter 产出<br>3. 所有 E2E 测试通过（SmolLM2, Gemma-4 E2B）<br>4. 新增模型只需写 YAML 模板 + 标量算子注册 | 🔴 待实现 |
| **REQ-UGS-004** | ModelConfig 缺失字段补全 | 补全 config.json 中未解析的模型配置字段 | 1. `use_double_wide_mlp` → 控制 FFN intermediate 倍率<br>2. `final_logit_softcapping` → 替代 executor 硬编码 30.0<br>3. `hidden_activation` → 自动选择 FfnActivation 枚举<br>4. 这些字段从 config.json → ModelConfig → ModelGeometry 全链路传递 | 🔴 待实现 |
