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
| **REQ-LOADER-011** | GGUF 格式支持 | 支持加载 GGUF 量化模型 (q4_0, q8_0) | 1. 使用 `gguf-rs` 解析<br>2. 能够加载 GGUF 张量到 Tensor 结构 | 🟢 已实现 (2026-02-05) [commit: bc1a030] |
| **REQ-LOADER-005** | ModelScope 支持 | 支持从魔搭社区(ModelScope)下载模型 | 1. `source` 配置项支持切换源<br>2. 自动处理 ModelScope 特有的文件结构 | 🟢 已实现 |
| **REQ-LOADER-006** | 动态模型发现 | 支持任意 HF Model ID，无需预注册 | 1. `Client::new("org/model-name", ModelKind::Chat)` 自动下载<br>2. 从 `config.json` 自动识别架构<br>3. 无需修改代码即可支持新模型 | 🟢 已实现 |
| **REQ-LOADER-007** | 架构自动识别 | 从模型配置文件自动推断架构类型 | 1. 读取 `config.json` 的 `model_type`/`architectures`<br>2. 匹配到对应的 Adapter<br>3. 支持常见架构的自动探测 | 🟢 已实现 |
| **REQ-LOADER-009** | Registry 清理 | 移除 KnownModel 枚举，实现纯动态加载 | 1. 移除 KnownModel<br>2. 移除硬编码 Repo 信息<br>3. 仅允许动态 Model ID | 🟢 已实现 |
| **REQ-LOADER-010** | Registry 删除与显式用途 | 彻底移除 Registry，API 显式指定 ModelKind | 1. 删除 Registry 与 ManifestOverride<br>2. `Client::new(model_id, kind)` 强制显式传入用途<br>3. 提供 `new_chat`/`new_embedding` 快捷方法<br>4. `manifest_from_config` 不再接受 overrides | 🟢 已实现 |
| **REQ-LOADER-012** | ONNX 格式支持 | 原生支持加载 .onnx 模型文件 (纯 Rust 实现) | 1. 集成官方完整 ONNX Proto 定义 (Enterprise Grade)<br>2. **禁止引入第三方推理引擎** (tract/ort)<br>3. 完整解析 Model/Graph/Node/Tensor 结构<br>4. 支持零拷贝/内存映射加载<br>5. **必须实现 Graph Pattern Matching**<br>6. **必须将子图映射为 Fused Kernels** | 🟢 已实现 (2026-02-05) [commit: 088b9a8] |
| **REQ-LOADER-013** | 自动格式探测 | 自动探测模型文件格式 (safetensors/GGUF/ONNX) | 1. 根据文件扩展名自动识别格式<br>2. 支持从 HF/MS 自动选择对应加载器<br>3. 无需用户手动指定格式 | 🟢 已实现 (2026-02-05) [commit: d16d3ea] |
| **REQ-LOADER-014** | GGUF 量化元数据读取 | 从 GGUF 文件读取量化类型信息 | 1. 从 GGUF 元数据读取 `general.quantization_version`<br>2. 从 GGUF tensor 信息读取实际量化类型 (Q4_0, Q8_0, Q5_K, etc.)<br>3. **禁止基于文件名推断** (Ω1: 真实性原则) | 🟢 已实现 (2026-02-07) [commit: 95c30d9] |
| **REQ-LOADER-015** | ONNX 精度元数据读取 | 从 ONNX 文件读取精度信息 | 1. 从 ONNX tensor dtype 读取实际精度 (fp32/fp16/int8/uint8)<br>2. **禁止基于文件名推断** (Ω1: 真实性原则) | 🟢 已实现 (2026-02-07) [commit: 95c30d9] |
| **REQ-LOADER-016** | 智能源选择 | HF 不可用时自动切换到 ModelScope | 1. HF 下载失败时自动尝试 ModelScope<br>2. 支持配置优先级 (HF→MS 或 MS→HF)<br>3. 记录源切换日志 | 🟢 已实现 (2026-02-05) [commit: d16d3ea] |
| **REQ-LOADER-017** | 统一加载入口 | 单一 API 支持所有格式和源 | `Loader::auto("repo/model")` 自动探测格式+源 | 🟢 已实现 (2026-02-05) [commit: d16d3ea] |
| **REQ-LOADER-018** | 迻除时模型热切换 | 支持在不重启进程的情况下切换模型 | 1. `client.swap_model(new_model)` API<br>2. 自动释放旧模型显存 (KV Cache & Weights)<br>3. 重新初始化新模型环境<br>4. 线程安全（阻塞新请求直到切换完成） | 🟢 已实现 (2026-02-07) [commit: HEAD] |
| **REQ-LOADER-019** | GGUF 架构元数据读取 | 从 GGUF 文件读取架构信息 | 1. 读取 GGUF 内置 `general.architecture` 字段 (如 "llama", "qwen2", "deepseek")<br>2. **禁止基于 Model ID 推断架构** (Ω1: 真实性原则)<br>3. 如果 GGUF 缺少架构元数据，返回明确的错误而非推测 | 🟢 已实现 (2026-02-07) [commit: 95c30d9] |
| **REQ-LOADER-020** | DeepSeek 架构支持 | 支持 DeepSeek V2/V3/R1 系列 MoE 模型 | 1. 实现 DeepSeekAdapter<br>2. 支持 MoE 架构 (671B 总参数, 37B 激活)<br>3. 从 config.json 识别 `model_type: "deepseek"`<br>4. 支持模型: DeepSeek-V3, DeepSeek-V2-Lite, DeepSeek-R1, **Kimi-K2** (使用 DeepSeek 架构)<br>5. 兼容 SafeTensors/GGUF/ONNX 格式 | 📋 待实现 |
| **REQ-LOADER-021** | 融合权重元数据驱动分割 | 融合权重 (如 QKV) 分割完全基于 config.json，禁止硬编码 | 1. `split_phi4_qkv` 等函数接收 `ModelConfig` 参数<br>2. Q 维度 = `config.hidden_size`<br>3. KV 维度 = `config.num_key_value_heads * config.head_dim`<br>4. **禁止**硬编码任何维度值 (如 3072, 1024)<br>5. 支持同一架构的不同变体 (不同 hidden_size / num_kv_heads)<br>6. **关联**: ARCH-QUANT-METADATA-001, ARCH-LOADER-FUSED-METADATA | 🟢 已实现 (2026-02-07) [commit: HEAD] |
| **REQ-LOADER-022** | 张量驱动配置推导 (Tensor-Driven) | 基于 Tensor Role Matching (Regex) 和张量形状推导配置，优先于硬编码逻辑 | 1. **核心**: 定义 `TensorRole` (Embedding/Attention/FFN)<br>2. **推导**: 纯张量形状推导 `hidden_size`, `num_heads`, `head_dim`<br>3. **禁止**: `if model == "llama"` 硬编码逻辑<br>4. **优先级**: 张量形状 > config.json | 🟢 已实现 (2026-02-08) |
| **REQ-LOADER-023** | 通用权重加载适配器 (Universal) | Loader 作为通用适配器，透明处理格式差异与精度转换 | 1. **F16->F32**: 后端需要时自动转换<br>2. **Universal**: 统一处理 SafeTensors/ONNX/GGUF<br>3. **Zero-Copy**: 仅在必要时转换，否则保持零拷贝<br>4. **关联**: ARCH-LOADER-003<br>5. **GGUF 量化分流**: 量化 tensor 存入 QuantizedTensor，native float 走 upload_native_tensor_with_convert | 🟢 已实现 (2026-02-08) |

### 2.1 GGUF 量化加载 (REQ-QUANT)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-QUANT-001** | GgmlDType→QuantType 桥接 | 映射 GGUF 量化类型到 gllm-kernels QuantType | `ggml_dtype_to_quant_type()` 覆盖 K-Quant/Classic/IQ 共 21 种类型 | 🟢 已实现 |
| **REQ-QUANT-002** | TensorProvider 量化元数据 | TensorProvider 暴露原始 GGML dtype | `ggml_dtype()` default method，GgufReader 实现返回实际 dtype | 🟢 已实现 |
| **REQ-QUANT-003** | QuantizedTensor 双存储 | WeightsHandle 支持量化/native 双存储 | `quantized` HashMap + `new_with_quantized()`/`quantized_tensor()`/`is_quantized()` | 🟢 已实现 |
| **REQ-QUANT-004** | upload_provider 量化分流 | 量化 tensor 跳过 GPU upload | 量化→QuantizedTensor，native float→upload_native_tensor_with_convert (F16/BF16→f32) | 🟢 已实现 |
| **REQ-QUANT-005** | TensorLookup 量化访问 | TensorLookup trait 支持量化 tensor 查询 | `get_quantized()` default method，WeightsHandle 实现 | 🟢 已实现 |
| **REQ-QUANT-006** | Backend quantized_matmul | Backend trait 量化矩阵乘法 | 按 QuantType 分发 kquant_matmul/classic_matmul/iq_matmul，CpuBackend 实现 | 🟢 已实现 |
| **REQ-QUANT-007** | Backend dequantize | Backend trait 反量化 | 24 种量化类型→f32，CpuBackend 实现 | 🟢 已实现 |

## 3. 核心功能 (REQ-CORE)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-CORE-001** | 自动后端检测 | 自动选择 CUDA/CPU (ROCm/Metal 计划中) | 1. `detect_backend()` 检测逻辑完整<br>2. 优先级: CUDA > ROCm > Metal > CPU<br>3. 未实现后端返回 `Unimplemented` | 🟢 已实现 (2026-02-07) [commit: 823e6bd] |
| **REQ-CORE-002** | 自动降级 | GPU OOM 时自动降级到 CPU | `FallbackEmbedder` 正常工作 | 🟢 已实现 |
| **REQ-CORE-003** | 量化支持 | 支持 Int4/Int8/AWQ/GPTQ/GGUF 加载 | 1. 能够加载并推理量化模型<br>2. GGUF Per-Tensor 混合精度加载 (QuantizedTensor)<br>3. Backend quantized_matmul/dequantize 分发 | 🟢 已实现 |
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
| **REQ-EXEC-003** | ONNX KV Cache 集成 | ONNX Generator 模型的 KV Cache 支持 | 1. 从 ONNX 图中提取 KV 输出张量<br>2. 跨轮缓存 KV 状态<br>3. 支持 PagedAttention 页面分配<br>4. 与 SwiftKV 蒸馏兼容 | 📋 待实现 |
| **REQ-SCHED-001** | PagedAttention 调度 | 实现自定义的分页注意力调度算法 (HGAL) | 1. 显存碎片率 < 5%<br>2. 支持动态 Block 分配<br>3. **禁止序列内页面分散换出**<br>4. **使用 LIRS 优先级计算** | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-002** | 双缓冲 KV Cache | 支持 GPU 双缓冲调度 (Swap 功能) | 1. Swap-in/Swap-out 已实现<br>2. **Warm-up 保护期机制**<br>3. **页面状态机 (Active/Standby/Swapped/Warm/Protected)** | 🟢 已实现 (2026-02-06) [commit: external-kernels] |
| **REQ-SCHED-003** | 动态批处理 | 支持 Continuous Batching | 1. 吞吐量优于 Static Batching<br>2. **序列完成自动移除**<br>3. **新序列动态加入**<br>4. **BatchAction 决策** (Continue/Complete/Pause)<br>5. **企业级死锁防护** (admit_waiting 无限循环修复) | 🟢 已实现 (2026-02-07) [commit: 823e6bd] |
| **REQ-SCHED-004** | Gang-Aware 调度 | 序列组整体调度，禁止序列内页面分散 | 1. **SequenceGroup 作为换出单位**<br>2. **All-or-nothing within one sequence**<br>3. 优先级调度 (FCFS/Priority) | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-005** | Cache Thrashing 防护 | 防止刚换入的页面立即被换出 | 1. **Warm-up 保护期** (默认 100ms)<br>2. **Thrash 率 < 1%**<br>3. 新换入页面不被选中为受害者 | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-006** | Working Set 检测 | 自动识别高频访问页面并锁定保护 | 1. **自动热页检测** (默认阈值 3 次访问)<br>2. **Protected 状态**<br>3. **保护解除机制** | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-007** | Chunked Prefill / SplitFuse | vLLM 2024 交织式混批优化 | **(已废弃)** 仅废弃 SplitFuse 混批路径；ChunkedConfig 以页面调度能力保留 | 🔴 已废弃 (由 REQ-SCHED-016 替代) |
| **REQ-SCHED-008** | SwiftKV 算法 | vLLM 2024 优化：KV Cache 压缩 | 1. **SingleInputKV**: 连续 N 个 KV 蒸馏为 1 个 (减少 50-75%)<br>2. **AcrossKV**: 跨层 KV 共享 (进一步减少 50%)<br>3. 精度损失 < 0.1% PPL<br>4. **JIT 兼容** (蒸馏在 CPU/Swap 时执行) | 🟢 已实现 (2026-02-02) [commit: 085bbf8] |
| **REQ-SCHED-009** | LMCache 跨请求共享 | 旧版 vLLM2024 LMCache 能力 | **(已废弃)** 由 `REQ-KV-001/002` 的 PrefixIndex + SessionKvCache 重构路径替代 | 🔴 已废弃 (Refactor 2026) |
| **REQ-SCHED-010** | LMCache 完全跳过前向计算 | 旧版 LMCache 命中跳过前向路径 | **(已废弃)** 由 `GlobalMemoryManager` 统一复用入口替代 | 🔴 已废弃 (Refactor 2026) |
| **REQ-SCHED-011** | SwiftKV CPU 蒸馏实现 | CPU 端真实 KV 蒸馏算法 | 1. SingleInputKV: 滑动窗口内聚合 KV<br>2. AcrossKV: 跨层余弦相似度计算<br>3. 精度验证: 蒸馏前后 PPL 差异 < 0.1%<br>4. 保持 CPU 端执行，兼容 JIT 统一路径 | 🟢 已实现 (2026-02-02) [commit: 0772fb1] |
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
| **REQ-KV-004** | KV 蒸馏泛型化 | SwiftKV distill 逻辑不得硬编码 `f32` | 1. `distill_cpu<E: Element>` 泛型接口<br>2. 相关相似度/评估接口同步泛型化<br>3. 禁止出现 `Vec<f32>` 固定签名作为核心实现 | 🟢 已实现 (2026-02-11) [commit: 8c41031] |

### 后续增强计划（未来版本）
| 优化 | 说明 | 状态 |
|------|------|------|
| **L3 分布式缓存** | Redis/NATS 等分布式后端支持 | 📋 未来计划 |
| **自适应 Chunk 大小** | 根据负载动态调整 chunk_size | 📋 未来计划 |
| **KV 增量更新** | 仅蒸馏变化的 KV 部分 | 📋 未来计划 |

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
| **REQ-TEST-005** | 功能模块覆盖 | 分层功能测试 | 1. Loader: 权重加载、格式转换<br>2. Inference: 生成、嵌入、重排序<br>3. Scheduler: PagedAttention、CB、Swap<br>4. Quantization: AWQ/GPTQ<br>5. Scheduler Refactor: PrefixIndex、SessionKvCache、KvPipeline、BatchOrderPolicy | 📋 待实现（随 REQ-SCHED-015~018 / REQ-KV-001~004 落地） |
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
| `tests/test_backend_compat.rs` | 后端一致性 (REQ-TEST-010) | 📋 待实现 |

## 6. 架构约束 (REQ-ARCH)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-ARCH-001** | 零拷贝推理 | 推理过程中 GPU 数据不回传 CPU | 符合 ARCH-GPU-001 | 🟢 已实现 |
| **REQ-ARCH-002** | 单一后端原则 | 全程在单一后端执行 | 符合 ARCH-SINGLE-BACKEND | 🟢 已实现 |
| **REQ-ARCH-003** | 纯 Rust 依赖原则 | 禁止引入 `candle`、`tch` 等重量级深度学习框架依赖 | 1. Cargo.toml 中无 candle/tch<br>2. 仅使用 safetensors/gguf-rs 等底层解析库<br>3. 计算核心完全自研 (gllm-kernels) | 🟢 已实现 (2026-02-05) [commit: fc36508] |

## 观测与调度 (REQ-OBS)

### REQ-OBS-001: SystemState 实时采集
- `BasicObserver::capture()` 必须从 scheduler/memory_manager/backend 实时采集所有指标
- `memory_pressure` 采集失败必须返回 `Err`，禁止 `unwrap_or(0.0)`
- `kv_fragmentation` 从 PagedScheduler 计算
- `waiting_queue_len` / `current_running_len` 从 ContinuousBatcher 读取
- **验收标准**: 构造已知状态，capture() 返回值与预期一致

### REQ-OBS-002: SchedulingPolicy 完整实现
- `AccuracyFirstPolicy` 使用 `memory_pressure` + `kv_fragmentation` + `waiting_queue_len` 三指标决策
- 新增 `BalancedPolicy` 变体
- `PolicyVariant` 枚举包含 Accuracy/Throughput/Balanced 三个变体
- **验收标准**: 各策略在不同 SystemState 输入下返回符合决策矩阵的 SchedulerDecision

### REQ-OBS-003: KernelStrategy 端到端传递
- `SchedulerDecision.kernel_strategy` 存入 `GeneratorForwardConfig`
- compat 层 forward 函数接收并记录 strategy
- **验收标准**: 设置 ThroughputFirst 策略后，forward_config 中 kernel_strategy 正确

### REQ-OBS-004: 策略热切换
- `Executor::set_policy(PolicyVariant)` 方法
- 下一个 `step()` 生效，不中断当前批次
- **验收标准**: 调用 set_policy 后下一次 step 使用新策略

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
