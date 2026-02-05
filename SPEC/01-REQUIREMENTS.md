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
| **REQ-LOADER-006** | 动态模型发现 | 支持任意 HF Model ID，无需预注册 | 1. `Client::new("org/model-name", ModelKind::Chat)` 自动下载<br>2. 从 `config.json` 自动识别架构<br>3. 无需修改代码即可支持新模型 | 🟢 已实现 |
| **REQ-LOADER-007** | 架构自动识别 | 从模型配置文件自动推断架构类型 | 1. 读取 `config.json` 的 `model_type`/`architectures`<br>2. 匹配到对应的 Adapter<br>3. 支持常见架构的自动探测 | 🟢 已实现 |
| **REQ-LOADER-008** | Manifest 降级为配置覆盖 | Manifest 仅用于特殊配置覆盖 | 1. 默认流程无需 Manifest<br>2. Manifest 仅用于 `rope_base` 等特殊覆盖<br>3. 新模型无需添加 Manifest 即可用 | 🟢 已实现 |
| **REQ-LOADER-009** | Registry 清理 | 移除 KnownModel 枚举，实现纯动态加载 | 1. 移除 KnownModel<br>2. 移除硬编码 Repo 信息<br>3. 仅允许动态 Model ID | 🟢 已实现 |
| **REQ-LOADER-010** | Registry 删除与显式用途 | 彻底移除 Registry，API 显式指定 ModelKind | 1. 删除 Registry 与 ManifestOverride<br>2. `Client::new(model_id, kind)` 强制显式传入用途<br>3. 提供 `new_chat`/`new_embedding` 快捷方法<br>4. `manifest_from_config` 不再接受 overrides | 🟢 已实现 |

### 重构说明 (REQ-LOADER-REFACTOR)

**当前问题**：
- 每个模型需要在 `KnownModel` 枚举中注册
- `hf_repo`、`model_scope_repo` 等信息硬编码
- 添加新模型需要修改代码并重新编译

**目标架构**：
```
用户输入: Client::new_chat("Qwen/Qwen3-0.6B")
  ↓
1. 构造 HF 下载地址: huggingface.co/Qwen/Qwen3-0.6B
  ↓
2. 下载模型文件 (config.json, tokenizer.json, safetensors)
  ↓
3. 读取 config.json → 识别架构 (Qwen3)
  ↓
4. 匹配 Qwen3Adapter → 加载权重
  ↓
5. (可选) 应用 Manifest 中的配置覆盖
```

**Manifest 重新定位**：
- **从**：模型注册表（每个模型都必须注册）
- **到**：配置覆盖层（仅用于特殊配置）
- **保留字段**：`rope_base_override`、`max_context_override`、`moe_config`
- **移除字段**：`hf_repo`、`model_scope_repo`、`aliases`（由用户输入直接指定）

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
| **REQ-SCHED-009** | LMCache 跨请求共享 | vLLM 2024 优化：L1/L2 KV Cache 架构 | 1. **L1 GPU** / **L2 CPU** 两层缓存<br>2. 相同提示命中时跳过 Prefill<br>3. 重复提示吞吐提升 10×+，命中率 > 70%<br>4. **AOT CUBIN 兼容** (使用现有 Memcpy Kernel) | 🟢 已实现 (2026-02-02) [commit: 085bbf8] |
| **REQ-SCHED-010** | LMCache 完全跳过前向计算 | 缓存命中时跳过 GPU 前向计算 | 1. 缓存命中时直接复用已有 KV handle<br>2. 跳过 embedding + attention + ffn 计算<br>3. 仅执行 sampling 生成第一个 token<br>4. 保持零拷贝原则 | 🟢 已实现 (2026-02-02) [commit: 0772fb1] |
| **REQ-SCHED-011** | SwiftKV CPU 蒸馏实现 | CPU 端真实 KV 蒸馏算法 | 1. SingleInputKV: 滑动窗口内聚合 KV<br>2. AcrossKV: 跨层余弦相似度计算<br>3. 精度验证: 蒸馏前后 PPL 差异 < 0.1%<br>4. 保持 CPU 端执行，兼容 AOT CUBIN | 🟢 已实现 (2026-02-02) [commit: 0772fb1] |

> **详细设计**: 见 [SPEC/02-ARCHITECTURE.md §2024 vLLM 优化](./02-ARCHITECTURE.md#2024-vllm-优化-arch-sched-2024)

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
| **后端** | `cpu`, `cuda` | ROCm/Metal 未来支持 |
| **模型类型** | `generator`, `embedding`, `rerank` | 三种核心功能 |
| **模型大小** | `mini` (最小) | 快速回归，CI 友好 |
| **功能模块** | `loader`, `inference`, `scheduler`, `quantization`, `vllm2024` | 分层验证 |

### 测试矩阵需求

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-TEST-001** | 后端覆盖测试 | CPU 和 CUDA 后端功能验证 | 1. 所有单元测试在 CPU 后端通过<br>2. 所有 E2E 测试在 CPU 后端通过<br>3. CUDA 后端可用时通过相同测试 | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-002** | Generator 模型矩阵 | 覆盖所有 Generator 模型类型 | 1. qwen3-7b (基准)<br>2. llama-4-8b (多模态)<br>3. phi-4-mini (轻量)<br>4. qwen3-moe (MoE)<br>5. smollm2-135m (超轻量)<br>6. smollm3-3b (轻量多用途)<br>7. internlm3-8b (高效推理) | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-003** | Embedding 模型矩阵 | 覆盖所有 Embedding 模型 | 1. qwen3-embed<br>2. bge-m3 (中文)<br>3. bge-m4 (升级版)<br>4. e5-small (轻量)<br>5. e5-base (标准)<br>6. e5-large (高精度)<br>7. m3e-base (中文)<br>8. jina-embeddings-v2-small (轻量英文)<br>9. jina-embeddings-v2-base (标准英文)<br>10. jina-embeddings-v4 (最新) | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-004** | Reranker 模型矩阵 | 覆盖所有 Reranker 模型 | 1. qwen3-rerank<br>2. bge-reranker-v2-m3 (轻量中文)<br>3. bge-rerank-v3 (升级版) | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-005** | 功能模块覆盖 | 分层功能测试 | 1. Loader: 权重加载、格式转换<br>2. Inference: 生成、嵌入、重排序<br>3. Scheduler: PagedAttention、CB、Swap<br>4. Quantization: AWQ/GPTQ<br>5. vllm2024: Chunked、SwiftKV、LMCache | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-006** | 量化格式测试 | 多种量化格式验证 | 1. AWQ (已实现)<br>2. GPTQ<br>3. SmoothQuant<br>4. 动态量化 | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-007** | 错误处理测试 | 边界条件和错误场景 | 1. OOM 处理<br>2. 无效输入<br>3. 权重损坏<br>4. 不支持的架构 | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-008** | 性能基准测试 | 吞吐量和延迟验证 | 1. Tokens/sec 吞吐量<br>2. 首token 延迟 (TTFT)<br>3. 内存占用<br>4. 性能回归检测 | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-009** | MoE 专项测试 | MoE 模型特殊验证 | 1. 专家路由正确性<br>2. 负载均衡<br>3. 动态专家选择 | 🟢 已实现 (2026-02-02) [commit: fc36508] |
| **REQ-TEST-010** | 后端一致性测试 | CPU vs CUDA 结果一致性 | 1. 相同输入产生相同输出<br>2. 数值精度在容差范围内 | 🟢 已实现 (2026-02-02) [commit: fc36508] |

### 测试文件规划

| 测试文件 | 覆盖维度 | 状态 |
|----------|---------|------|
| `tests/test_model_matrix.rs` | 模型矩阵 (REQ-TEST-002/003/004) | 🔵 新增 |
| `tests/test_backend_compat.rs` | 后端一致性 (REQ-TEST-010) | 🔵 新增 |
| `tests/test_quantization.rs` | 量化格式 (REQ-TEST-006) | 🔵 新增 |
| `tests/test_error_handling.rs` | 错误处理 (REQ-TEST-007) | 🔵 新增 |
| `tests/test_moe_routing.rs` | MoE 专项 (REQ-TEST-009) | 🔵 新增 |
| `tests/test_performance.rs` | 性能基准 (REQ-TEST-008) | 🔵 新增 |

## 6. 架构约束 (REQ-ARCH)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-ARCH-001** | 零拷贝推理 | 推理过程中 GPU 数据不回传 CPU | 符合 ARCH-GPU-001 | 🟢 已实现 |
| **REQ-ARCH-002** | 单一后端原则 | 全程在单一后端执行 | 符合 ARCH-SINGLE-BACKEND | 🟢 已实现 |
