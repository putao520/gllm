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
| **REQ-LOADER-023** | 通用权重加载适配器 ⛔ (Arch Vetoed) | Loader 层统一量化处理，必须封杀任何 F16->F32 运行时隐式升格策略。 | 1. **Zero-Copy**: 强行根据 `TurboQuantBits` 将权重对齐至 JIT 内核接受的极化格式<br>2. **Universal**: 统一处理 SafeTensors/ONNX/GGUF<br>3. **Float Annihilation**: 所有 Native Float 权重必须在装载期一并压缩至静态位宽，无浮点驻留缓冲 | 🔴 架构接管重构中 |

### 2.1 GGUF 量化加载 (REQ-QUANT)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-QUANT-001** | GgmlDType→QuantType 桥接 | 映射 GGUF 量化类型到 gllm-kernels QuantType | `ggml_dtype_to_quant_type()` 覆盖 K-Quant/Classic/IQ 共 21 种类型 | 🟢 已实现 |
| **REQ-QUANT-002** | TensorProvider 量化元数据 | TensorProvider 暴露原始 GGML dtype | `ggml_dtype()` default method，GgufReader 实现返回实际 dtype | 🟢 已实现 |
| **REQ-QUANT-003** | QuantizedTensor 双存储 | WeightsHandle 支持量化/native 双存储 | `quantized` HashMap + `new_with_quantized()`/`quantized_tensor()`/`is_quantized()` | 🟢 已实现 |
| **REQ-QUANT-004** | upload_provider 量化分流 | 量化 tensor 跳过 GPU upload | 量化→QuantizedTensor，非量化强制位宽压缩→upload_compressed_tensor_with_alignment () | 🟢 已实现 |
| **REQ-QUANT-005** | TensorLookup 量化访问 | TensorLookup trait 支持量化 tensor 查询 | `get_quantized()` default method，WeightsHandle 实现 | 🟢 已实现 |
| **REQ-QUANT-006** | Backend quantized_matmul | Backend trait 量化矩阵乘法 | JIT 编译静态位宽向量操作，完全抹除运行时 dispatch | 🟢 已实现 |
| **REQ-QUANT-007** | Backend dequantize | (已废除) Backend trait 反量化 | 严禁运行时逆向量化至 f32，强制使用 TurboQuant 静态整型/定点累加器 | 🚫 架构否决 |

## 3. 核心功能 (REQ-CORE)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-CORE-001** | 自动后端检测 | 自动选择 CUDA/CPU (ROCm/Metal 计划中) | 1. `detect_backend()` 检测逻辑完整<br>2. 优先级: CUDA > ROCm > Metal > CPU<br>3. 未实现后端返回 `Unimplemented` | 🟢 已实现 (2026-02-07) [commit: 823e6bd] |
| **REQ-CORE-002** | 任务级 OOM 降级 | GPU OOM 时整个任务自动退回 CPU (禁止引发精度截断的妥协) | `FallbackEmbedder` 正常工作 | 🟢 已实现 |
| **REQ-CORE-003** | 静态极化量化支持 (TurboQuant) | 支持统一强转极化格式 (INT4/FP8) | 1. 能够加载量化模型权重<br>2. 运行时消除多态混合精度分派，全网强制统一至 TurboQuant 固定块规格<br>3. 废除反量化，只提供定点/微浮点硬派算子 | 🟢 已实现 |
| **REQ-CORE-004** | 精度优先架构 | 系统强制运行在"精度优先"模式 | 1. 强制启用 Deterministic Scheduling<br>2. 强制启用 Phase Isolation<br>3. **移除** 任何吞吐量优先的妥协配置 | 🟢 已实现 (2026-02-07) [commit: 823e6bd] |

## 4. 高级调度与内存管理 (REQ-SCHED)

> **详细设计**: 见 [SPEC/DOCS/scheduling/hgal-scheduler-algorithm.md](./DOCS/scheduling/hgal-scheduler-algorithm.md)
> **架构原则**: Accuracy First (见 ARCH-ACCURACY)。**所有牺牲精度的吞吐量优化均被废弃。**

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-SCHED-012** | 确定性调度 (Deterministic) | 强制 Batch 内请求排序，消除浮点非结合性误差 | 1. `ContinuousBatcher` 输出严格按 ID 排序<br>2. 相同输入在不同负载下输出比特级一致<br>3. **禁止** 随机插入插槽 | 🟢 已实现 (2026-02-06) [commit: HEAD] |
| **REQ-SCHED-013** | 阶段隔离 (Phase Isolation) | 严禁 Prefill 和 Decode 请求混合在同一 Batch | 1. Batch 状态机：纯 Prefill 或 纯 Decode<br>2. 允许 Prefill 内 ChunkedConfig 分块，不允许 Prefill/Decode 混批<br>3. 消除计算图抖动 | 🟢 已实现 (2026-02-06) [commit: HEAD] |
| **REQ-EXEC-001** | 串行微批次执行 | Executor 内部串行执行 Batch 中的请求 | 1. `step()` 循环内串行调用 `forward`<br>2. 避免 GPU 并行规约误差<br>3. **无配置项**，强制开启 | 🟢 已实现 (2026-02-06) [commit: HEAD] |
| **REQ-EXEC-002** | ONNX 推理执行引擎 | 实现 ONNX 模型的完整推理执行能力 | 1. **Embedding/Reranker**: 单次前向传播已支持 (BERT/XLM-R/Qwen3)<br>2. **Generator**: 实现生成循环 + KV Cache + Sampling<br>3. 使用 FusedKernel 执行 ONNX 子图<br>4. 支持 ONNX 模型的动态批处理<br>5. 与现有 PagedAttention 调度器集成 | 🟢 已实现 (2026-02-07) [commit: 3a0957d] |
| **REQ-EXEC-003** | ONNX KV Cache 集成 | ONNX Generator 模型的 KV Cache 支持 | 1. 从 ONNX 图中提取 KV 输出张量<br>2. 跨轮缓存 KV 状态<br>3. 支持 PagedAttention 页面分配 | 🟢 已实现 (2026-03-15) |
| **REQ-SCHED-001** | PagedAttention 调度 | 实现自定义的分页注意力调度算法 (HGAL) | 1. 显存碎片率 < 5%<br>2. 支持动态 Block 分配<br>3. **禁止序列内页面分散换出**<br>4. **使用 LIRS 优先级计算** | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-002** | 双缓冲 KV Cache | 支持 GPU 双缓冲调度 (Swap 功能) | 1. Swap-in/Swap-out 已实现<br>2. **Warm-up 保护期机制**<br>3. **页面状态机 (Active/Standby/Swapped/Warm/Protected)** | 🟢 已实现 (2026-02-06) [commit: external-kernels] |
| **REQ-SCHED-003** | 动态批处理 | 支持 Continuous Batching | 1. 吞吐量优于 Static Batching<br>2. **序列完成自动移除**<br>3. **新序列动态加入**<br>4. **BatchAction 决策** (Continue/Complete/Pause)<br>5. **企业级死锁防护** (admit_waiting 无限循环修复) | 🟢 已实现 (2026-02-07) [commit: 823e6bd] |
| **REQ-SCHED-004** | Gang-Aware 调度 | 序列组整体调度，禁止序列内页面分散 | 1. **SequenceGroup 作为换出单位**<br>2. **All-or-nothing within one sequence**<br>3. 优先级调度 (FCFS/Priority) | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-005** | Cache Thrashing 防护 | 防止刚换入的页面立即被换出 | 1. **Warm-up 保护期** (默认 100ms)<br>2. **Thrash 率 < 1%**<br>3. 新换入页面不被选中为受害者 | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-006** | Working Set 检测 | 自动识别高频访问页面并锁定保护 | 1. **自动热页检测** (默认阈值 3 次访问)<br>2. **Protected 状态**<br>3. **保护解除机制** | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-007** | Chunked Prefill / SplitFuse | vLLM 2024 交织式混批优化 | **(已废弃)** 仅废弃 SplitFuse 混批路径；ChunkedConfig 以页面调度能力保留 | 🔴 已废弃 (由 REQ-SCHED-016 替代) |
| **REQ-SCHED-008** | SwiftKV 算法 | vLLM 2024 优化：KV Cache 压缩 | **(已废弃)** 违宪：禁止 CPU 端或生成循环中修改 KV 数据（详见 ARCH 约束） | 🔴 违宪 (Vetoed 2026) |
| **REQ-SCHED-009** | LMCache 跨请求共享 | 旧版 vLLM2024 LMCache 能力 | **(已废弃)** 由 `REQ-KV-001/002` 的 PrefixIndex + SessionKvCache 重构路径替代 | 🔴 已废弃 (Refactor 2026) |
| **REQ-SCHED-010** | LMCache 完全跳过前向计算 | 旧版 LMCache 命中跳过前向路径 | **(已废弃)** 由 `GlobalMemoryManager` 统一复用入口替代 | 🔴 已废弃 (Refactor 2026) |
| **REQ-SCHED-011** | SwiftKV CPU 蒸馏实现 | CPU 端真实 KV 蒸馏算法 | **(已废弃)** 违宪：禁止在 CPU 端处理浮点张量及其相似度计算 | 🔴 违宪 (Vetoed 2026) |
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
| **REQ-KV-004** | KV 蒸馏泛型化 | SwiftKV distill 逻辑不得硬编码 `f32` | **(已废弃)** 违宪：任何形式的 KV 蒸馏重算均已禁止 | 🔴 违宪 (Vetoed 2026) |

### 4.2 自适应 Chunk 与 KV 增量更新 (REQ-KV-EXT)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-KV-EXT-001** | 自适应 Chunk 大小 | 根据运行时负载（L1 可用页数、并发请求数、prompt 长度）动态调整 prefill chunk_size，替代当前硬编码 max_seq_len | 1. `AdaptiveChunkPolicy` 结构体，输入 L1 可用页数/并发请求数/prompt 长度，输出 chunk_size<br>2. Executor 调用 `plan_prefill()` 时使用自适应 chunk_size 而非 `max_seq_len`<br>3. chunk_size 范围 `[ChunkedConfig::chunk_size, max_seq_len]`，下界为 ChunkedConfig 默认值 64<br>4. 高负载（L1 可用 < 25%）时 chunk_size 缩小至下界<br>5. 低负载（L1 可用 > 75%）时 chunk_size 扩大至 max_seq_len<br>6. 单元测试覆盖高/低/中三种负载场景 | 🟢 已实现 (2026-03-15) |
| **REQ-KV-EXT-002** | KV 增量蒸馏 | SwiftKV distill 仅处理自上次蒸馏以来变化的 KV 页面，避免全量重算 | 1. `SwiftKvState` 新增 `last_distilled_page: usize` 追踪上次蒸馏边界<br>2. `distill_cpu_incremental()` 仅处理 `[last_distilled_page..]` 范围的页面<br>3. 增量蒸馏结果与全量蒸馏数值一致（容差 < 1e-6）<br>4. 跨轮次 session 复用时正确维护蒸馏边界<br>5. `prepare_next_turn()` 重置 Working 管线蒸馏边界，保留 Conversation 管线<br>6. 单元测试验证增量 vs 全量一致性 | 🔴 违宪 (Vetoed 2026) |

### 4.3 JIT 缓存协议规范 (REQ-JIT-CACHE)

> **关联规范**: [SPEC/DOCS/scheduling/jit-cache-protocol.md](./DOCS/scheduling/jit-cache-protocol.md)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-JIT-CACHE-001** | 全核心态级预编译 | 所有算子必须在模型加载阶段完成全网计算图预编译 | 1. 禁用独立算子的 JIT 支持<br>2. 仅允许全模型作为一个整体计算图被缓存 (`GraphExecutor`)<br>3. 推理热路径中无任何单层或单算子重编译 | 🟢 已设定 |
| **REQ-JIT-CACHE-002** | CPU/GPU 算子与流程统一 | CPU 后端与 GPU 后端的 JIT 缓存行为必须 100% 对齐 | 1. **ARCH-CPU-GPU-UNIFIED**: 删除 CPU 特有算子图节点和变体<br>2. 共同使用 `build_fused_attention_layer_graph_symbolic`<br>3. 动态 seq_len 使用 `SymDim::Symbolic` | 🟢 已设定 |

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
- 系统被设定为对所有有效指令强制锁定 **TurboQuant (W4A4/W8A8)** 运行。
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

### REQ-ERR-002: OOM Fallback 显式化
- GPU→CPU fallback 必须 `log::warn!` 记录
- 返回 `FallbackResult<T>` 携带 `fallback_used: bool` 标记
- **验收标准**: 触发 OOM fallback 后返回值 `fallback_used == true`

### REQ-ERR-003: Backend Detection 错误传播
- `detection.rs` 的 `expect()` 替换为 `Result` 返回
- 探测失败返回 `Err(BackendContextError)`，不 panic
- **验收标准**: 后端探测失败时返回 Err 而非 panic

## gllm-kernels 侧需求 (REQ-KERNELS)

> **仓库**: `/home/putao/code/rust/gllm-kernels/`
> **关联**: 以下需求在 gllm-kernels 仓库实现，gllm 侧仅记录需求定义和验收标准

### 8.1 IQ Codebook 与 TurboQuant 映射 (REQ-KERNELS-IQ)

> 🚨 **架构合规性警报 (Architect Veto)**: 严禁在 Kernel 运行时进行动态的 `dequant -> f32` 反量化。所有 IQ 权重的解码均被静态化至 Mega-Kernel 内联的整型/定点逻辑中。所有的 `dequant` 后缀将被废除或视为零开销的指令级特征投影。

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

> **背景**: 当前存在三个相互关联的问题：(1) `CompilerGraph` shape 在构建时硬编码为具体数值，`total_seq` 每步 +1 导致 CachedGQA 图每步重编译；(2) YAML 模板与实际执行路径断层，每新增模型都需在 `decoder_forward.rs` 手写分支。GPT-2 架构已于 2026-03-19 从支持列表中移除（Latest Version Only 策略）。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-JIT-GRAPH-001** | JIT 图 Symbolic Shape 支持 | `CompilerGraph` 支持 symbolic 维度，运行时传入具体值，避免 shape 变化时重编译 | 1. `CompilerGraph` 支持 `SymDim`（符号维度），可声明 `Concrete(usize)` 或 `Symbolic(String)`<br>2. `total_seq`、`batch_size` 等动态维度必须声明为 `SymDim::Symbolic`<br>3. 编译一次，运行时通过 shape binding 传入具体值<br>4. CachedGQA 图不再每 decode step 重编译<br>5. 禁止在图构建时将动态维度硬编码为具体数值 | 🟢 已实现 |
| **REQ-JIT-GRAPH-002** | GPT-2 路径 JIT 化 | ~~`gpt2_forward_sequence()` 中所有 GEMM 必须走 JIT `CompilerGraph`~~ | ~~已实现后废弃~~ | ⚠️ 已废弃 (GPT-2 于 2026-03-19 移除) |
| **REQ-JIT-GRAPH-003** | 图执行器打通（YAML → JIT 端到端） | `OnnxGraph`（从 YAML 展开）直接驱动 JIT 执行，消除 `decoder_forward.rs` 手写分支 | 1. `src/graph/executor.rs` 的 `FusedGraph` 执行器能执行完整 decoder forward（含 KV cache）<br>2. 新模型只需提供 YAML 文件，不需要修改任何 Rust 代码<br>3. YAML → `OnnxGraph` → 图优化 → JIT 编译 → 执行链路端到端跑通<br>4. 现有手写分支（GPT-2、MoE、GQA）逐步迁移到图执行器<br>5. 图执行器支持 symbolic shape binding（依赖 REQ-JIT-GRAPH-001） | 🟢 已实现 |

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
