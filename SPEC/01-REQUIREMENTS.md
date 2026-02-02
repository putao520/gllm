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
| **REQ-LOADER-002** | 智能缓存系统 | 支持本地缓存管理，避免重复下载 | `~/.gllm/cache` 结构正确，支持校验和验证 | 🟢 已实现 |
| **REQ-LOADER-003** | 并发加载 | 支持多线程并发下载和张量加载 | 大模型加载速度显著提升 | 🟢 已实现 |
| **REQ-LOADER-004** | 权重转换 | 支持 safetensors/bin 自动探测与转换 | 兼容主流权重格式 | 🟢 已实现 |
| **REQ-LOADER-005** | ModelScope 支持 | 支持从魔搭社区(ModelScope)下载模型 | 1. `source` 配置项支持切换源<br>2. 自动处理 ModelScope 特有的文件结构 | 🟢 已实现 |

## 3. 核心功能 (REQ-CORE)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-CORE-001** | 自动后端检测 | 自动选择 CUDA/CPU (ROCm/Metal 计划中) | `detect_backend()` 返回正确类型 | 🟡 部分实现 |
| **REQ-CORE-002** | 自动降级 | GPU OOM 时自动降级到 CPU | `FallbackEmbedder` 正常工作 | 🟢 已实现 |
| **REQ-CORE-003** | 量化支持 | 支持 Int4/Int8/AWQ/GPTQ/GGUF 加载 | 能够加载并推理量化模型 | 🟢 已实现 |

## 4. 高级调度与内存管理 (REQ-SCHED)

> **详细设计**: 见 [SPEC/DOCS/scheduling/hgal-scheduler-algorithm.md](./DOCS/scheduling/hgal-scheduler-algorithm.md)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-SCHED-001** | PagedAttention 调度 | 实现自定义的分页注意力调度算法 (HGAL) | 1. 显存碎片率 < 5%<br>2. 支持动态 Block 分配<br>3. **禁止序列内页面分散换出**<br>4. **使用 LIRS 优先级计算** | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-002** | 双缓冲 KV Cache | 支持 GPU 双缓冲调度 (Swap 功能) | 1. Swap-in/Swap-out 已实现<br>2. **Warm-up 保护期机制**<br>3. **页面状态机 (Active/Standby/Swapped/Warm/Protected)** | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-003** | 动态批处理 | 支持 Continuous Batching | 1. 吞吐量优于 Static Batching<br>2. **序列完成自动移除**<br>3. **新序列动态加入**<br>4. **BatchAction 决策** (Continue/Complete/Pause) | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-004** | Gang-Aware 调度 | 序列组整体调度，禁止序列内页面分散 | 1. **SequenceGroup 作为换出单位**<br>2. **All-or-nothing within one sequence**<br>3. 优先级调度 (FCFS/Priority) | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-005** | Cache Thrashing 防护 | 防止刚换入的页面立即被换出 | 1. **Warm-up 保护期** (默认 100ms)<br>2. **Thrash 率 < 1%**<br>3. 新换入页面不被选中为受害者 | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-006** | Working Set 检测 | 自动识别高频访问页面并锁定保护 | 1. **自动热页检测** (默认阈值 3 次访问)<br>2. **Protected 状态**<br>3. **保护解除机制** | 🟢 已实现 (2026-02-02) [commit: 063f150] |
| **REQ-SCHED-007** | Chunked Prefill / SplitFuse | vLLM 2024 优化：消除 Prefill-Decode 阶段隔离 | 1. **Chunked Prefill**: Prefill 请求切分为 Chunk，与 Decode 交织调度<br>2. **SplitFuse**: Q/K/V 分离计算 + 融合 Attention<br>3. Tail Latency (P99) 降低 30-50%<br>4. **AOT CUBIN 兼容** (纯调度优化，无需新 Kernel) | 🟢 已实现 (2026-02-02) [commit: 085bbf8] |
| **REQ-SCHED-008** | SwiftKV 算法 | vLLM 2024 优化：KV Cache 压缩 | 1. **SingleInputKV**: 连续 N 个 KV 蒸馏为 1 个 (减少 50-75%)<br>2. **AcrossKV**: 跨层 KV 共享 (进一步减少 50%)<br>3. 精度损失 < 0.1% PPL<br>4. **AOT CUBIN 兼容** (蒸馏在 CPU/Swap 时执行) | 🟢 已实现 (2026-02-02) [commit: 085bbf8] |
| **REQ-SCHED-009** | LMCache 跨请求共享 | vLLM 2024 优化：三层 KV Cache 架构 | 1. **L1 GPU** / **L2 CPU** / **L3 分布式** 三层缓存<br>2. 相同提示命中时跳过 Prefill<br>3. 重复提示吞吐提升 10×+，命中率 > 70%<br>4. **AOT CUBIN 兼容** (使用现有 Memcpy Kernel) | 🟢 已实现 (2026-02-02) [commit: 085bbf8] |

> **详细设计**: 见 [SPEC/02-ARCHITECTURE.md §2024 vLLM 优化](./02-ARCHITECTURE.md#2024-vllm-优化-arch-sched-2024)

## 5. 架构约束 (REQ-ARCH)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-ARCH-001** | 零拷贝推理 | 推理过程中 GPU 数据不回传 CPU | 符合 ARCH-GPU-001 | 🟢 已实现 |
| **REQ-ARCH-002** | 单一后端原则 | 全程在单一后端执行 | 符合 ARCH-SINGLE-BACKEND | 🟢 已实现 |
