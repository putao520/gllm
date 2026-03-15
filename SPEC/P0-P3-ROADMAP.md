# gllm P0-P3 全优先级工作路线图

> **生成日期**: 2026-03-14
> **基于**: 全量 SPEC 审计 + 代码实现状态深度审计
> **SSOT**: 本文档是 P0-P3 工作项的唯一规划源

---

## 审计修正声明

前次审计对以下模块的判断有误，此处修正：

| 模块 | 前次判断 | 实际状态 |
|------|---------|---------|
| Generation Loop | "不完整/builder only" | ✅ 完整实现：`Executor::step()` 实现了 token-by-token 生成循环 |
| Token Sampling | "0% 未实现" | ✅ 完整实现：temperature/top-k/top-p/greedy 全部实现 |
| Scheduler→Executor 集成 | "断裂" | ✅ 已集成：`ContinuousBatcher` + `PagedScheduler` 驱动 `step()` |
| KV Cache 分配 | "未接入" | ✅ 已接入：`alloc_kv_cache()` + `advance()` + session 复用 |

**真正的 P0 瓶颈**：KV Cache 增量持久化（当前每步重算全部 K/V，未写入 cache buffer）。

---

## P0 — 性能关键路径（KV Cache 增量化 + 错误处理）

### P0-1: KV Cache 增量持久化 (REQ-KV-005, ARCH-KV-PERSIST) ✅ 已完成

**完成状态**: CPU 路径 — `build_kv_projection_graph` / `execute_kv_projection` / `write_kv_to_cache` 实现 JIT KV projection。GPU 路径 — CUDA/HIP/Metal 三后端均支持 `is_incremental` 分支（GPU proj → dtoh → KV write → CPU attn → GPU post-attn）。`gpu_write_kv_cache()` 共享函数消除三后端重复代码。

### P0-2: 消除静默失败 (REQ-ERR-001) ✅ 已完成

**完成状态**: `Err(_)` catch-all 为 0 匹配。`let _ =` 全部为合法参数抑制或有意忽略（已审计）。`unwrap_or(0.0)` 在 fallback.rs rerank_pair 处已添加 `log::warn!`。

### P0-3: OOM Fallback 显式化 (REQ-ERR-002) ✅ 已完成

**完成状态**: `FallbackResult<T>` 携带 `fallback_used: bool` 已实现。所有 fallback 触发点（OomFallback::run, FallbackGenerator, FallbackEmbedder, FallbackReranker）均有 `log::warn!`。

### P0-4: Backend Detection 错误传播 (REQ-ERR-003) ✅ 已完成

**完成状态**: `detect_backend()` / `detect_backend_generic()` / `detect_f32()` 全部返回 `Result<DetectedBackend, BackendError>`。无 `expect()` 或 `unwrap()` 调用。

---

## P1 — 架构模板扩展 + 观测性完善

### P1-1: 架构模板补全（11 个缺失架构） ✅ 已完成

**问题**: 仅 Qwen3 和 Llama 有 YAML 模板，其余 11 个架构只在 manifest 中识别。

**完成状态**: 9 个 YAML 模板全部就绪（qwen3, llama, mistral3, glm4, phi4, gemma2, gpt2next, xlmr, deepseek），6 个架构映射到已有模板（Ministral→mistral3, GLM5→glm4, XlmRNext→xlmr, Qwen3MoE→qwen3, SmolLM2→llama, InternLM3→llama）。

**按优先级分批**:

| 批次 | 架构 | 理由 |
|------|------|------|
| P1-1a | Mistral3, Ministral | SUPPORTED_MODELS.md 列出，用户需求高 |
| P1-1b | GLM4, GLM5 | 中文生态重要模型 |
| P1-1c | Phi4 | 轻量级端侧模型 |
| P1-1d | Gemma2 | Google 生态 |
| P1-1e | GPT2Next (GPT-OSS) | OpenAI 开源模型 |
| P1-1f | XlmR, XlmRNext | Embedding/Reranker 架构（BERT-like） |
| P1-1g | DeepSeek (REQ-LOADER-020) ✅ | MoE 671B，DeepSeekAdapter + MoEConfig 元数据提取 |

**每个模板需要**:
1. `src/arch/templates/{arch}.yaml` — 层定义
2. `src/arch/registry.rs` — 注册映射
3. E2E 测试验证

### P1-2: Observer Phase 2 指标采集 (07-OBSERVABILITY §2, §7 Phase 2) ✅ 已完成

**完成状态**: `logits_entropy` 采集已实现（Shannon 熵计算 + batch 平均）。`attention_sparsity` 设为 0.0（MHA op 当前仅输出 attn_out，需扩展才能采集注意力权重）。3 个单元测试覆盖。

### P1-3: KernelStrategy 端到端传递 (REQ-OBS-003) ✅ 已完成

**完成状态**: `SchedulerDecision.kernel_strategy` 在 `step()` 中写入 `forward_config.kernel_strategy`，CPU backend 已接收并记录 strategy。所有 Policy 变体（AccuracyFirst/ThroughputFirst/Balanced）正确设置 strategy。

### P1-4: 策略热切换 (REQ-OBS-004, 07-OBSERVABILITY §4) ✅ 已完成

**完成状态**: `Executor::set_policy(PolicyVariant)` 已实现，下一个 `step()` 立即生效，无需重启。

### P1-5: 量化推理加速路径 ✅ 已完成

**完成状态**: `quantized_linear()` 已在 forward 路径中调用，检测量化权重时直接走 `Backend::quantized_matmul()` 而非先 dequant。

---

## P2 — 测试补全 + 代码质量

### P2-1: 后端一致性测试 (REQ-TEST-010) ✅ 已完成

**文件**: `tests/test_backend_compat.rs` — 6 个测试覆盖 BackendError Display、KvCacheHandle 相等性/哈希、std::error::Error 实现、CPU 确定性 embedding/generation。

### P2-2: 功能模块覆盖测试 (REQ-TEST-005) ✅ 已完成

**文件**: `tests/test_scheduler_refactor.rs` — 15 个测试覆盖 PrefixIndex（插入/查找/部分匹配/追加复用/无匹配/空查询）、BatchOrderPolicy、KvPipeline、SequenceGroup。

### P2-3: 跨语言对齐测试 (REQ-TEST-011) ✅ 已完成

**目录**: `tests/e2e_alignment/` — generate_golden.py（PyTorch 基准生成）、requirements.txt、README.md。`tests/e2e_alignment.rs` — Rust 对齐测试（`#[ignore]` 标记，需先生成 golden data）。容差: FP32 < 1e-5, cosine > 0.9999。

### P2-4: TEST-XXX 注释补全 ✅ 已完成

**完成状态**: 15 个测试文件补充了 SPEC 06-TESTING-STRATEGY §1.2 格式注释，共 74 条 TEST-XXX 注释覆盖 21 个测试文件。格式: `TEST-{TYPE}-{SEQ}` + 关联需求 + 测试类型 + 期望结果。

### P2-5: IQ 系列 Codebook 嵌入

**问题**: IQ1S/IQ1M/IQ2XXS/IQ2XS/IQ2S/IQ3XXS/IQ3S 在 gllm-kernels 中输出 zeros（codebook 未嵌入）。

**修改范围**: gllm-kernels `src/backend/mod.rs` — 嵌入 E8/D4 lattice codebook 表。

**注意**: 这是 gllm-kernels 侧修改，不在 gllm 仓库内。

---

## P3 — 未来增强

### P3-1: MoE 路由执行 (REQ-MODEL-004) ✅ 已完成

**完成状态**: Scalar MoE 路由已在 executor forward 路径中实现（专家选择 + top-k gating）。

### P3-2: Thinking Head 提取 ✅ 已完成

**完成状态**: `split_thinking_content()` 已实现，支持 thinking token 识别与分离。

### P3-3: PyTorch 格式支持 ✅ 已完成

**完成状态**: `pytorch.rs` 纯 Rust 实现 pickle 反序列化 + safetensors 转换（内联最小化 pickle 协议解析器，无 candle/tch 依赖，符合 REQ-ARCH-003）。`Loader::load()` 的 `WeightFormat::PyTorch` 分支调用 `convert_bins_to_safetensors()` 转换后走标准 safetensors 加载路径。默认启用，无需 feature flag。

### P3-4: GPU TileLevelFusion / ComputeRoot

**问题**: 这两种融合模式在 GPU codegen 中返回 error。

**修改范围**: gllm-kernels GPU codegen 扩展。

### P3-5: GLLM_CACHE_DIR 环境变量 (ARCH-MODEL-CACHE) ✅ 已完成

**完成状态**: `loader/mod.rs:92` 已实现 `GLLM_CACHE_DIR` 环境变量读取，覆盖默认缓存路径。

### P3-6: 分布式 KV Cache (L3)

**问题**: Redis/NATS 等分布式后端支持。

**状态**: 未来计划，暂不排期。

---

## 优先级总览

```
P0 (性能关键 + 正确性)     P1 (功能完善)           P2 (质量)              P3 (未来)
─────────────────────     ──────────────────     ──────────────────     ──────────────
P0-1 KV Cache 增量化 ✅    P1-1 架构模板 ×11 ✅   P2-1 后端一致性测试 ✅  P3-1 MoE 路由 ✅
P0-2 消除静默失败 ✅       P1-2 Observer Phase2 ✅ P2-2 调度器重构测试 ✅  P3-2 Thinking Head ✅
P0-3 OOM Fallback ✅       P1-3 KernelStrategy ✅  P2-3 跨语言对齐测试 ✅  P3-3 PyTorch 格式 ✅
P0-4 Backend Detection ✅  P1-4 策略热切换 ✅      P2-4 TEST-XXX 注释 ✅  P3-4 GPU 融合扩展
                           P1-5 量化推理加速 ✅    P2-5 IQ Codebook       P3-5 GLLM_CACHE_DIR ✅
                                                                         P3-6 分布式 KV Cache
```

## 依赖关系

```
P0-1 (KV Cache 增量化) ← 无外部依赖，可立即开始
P0-2/3/4 (错误处理) ← 无依赖，可并行
P1-1 (架构模板) ← 无依赖，可并行
P1-2 (Observer Phase2) ← 依赖 P0-1（需要 forward 路径中的 logits/attention 数据）
P1-3 (KernelStrategy) ← 无依赖
P1-4 (策略热切换) ← 依赖 P1-3
P1-5 (量化加速) ← 依赖 P0-1（需要 KV cache 正确工作）
P2-1 (后端一致性) ← 依赖 P0-1
P2-5 (IQ Codebook) ← gllm-kernels 侧修改
P3-1 (MoE) ← 依赖 P1-1g (DeepSeek 模板) + gllm-kernels MoE kernel
```

## 修改文件清单

| 优先级 | 文件 | 修改内容 |
|--------|------|---------|
| P0-1 | `src/compat/decoder_forward.rs` | KV cache 增量持久化 |
| P0-1 | `src/compat/gpu_compile.rs` | GPU 路径 KV cache 同步 |
| P0-2 | `src/` 全局 | 消除 `let _ =` / `Err(_)` / `unwrap_or` |
| P0-3 | `src/backend/fallback.rs` | FallbackResult + log::warn |
| P0-4 | `src/backend/detection.rs` | expect → Result |
| P1-1 | `src/arch/templates/*.yaml` + `registry.rs` | 11 个架构模板 |
| P1-2 | `src/engine/executor.rs` | logits_entropy / attention_sparsity 采集 |
| P1-3 | `src/engine/executor.rs` | kernel_strategy 传递 |
| P1-4 | `src/engine/executor.rs` | set_policy() 方法 |
| P1-5 | `src/compat/decoder_forward.rs` | quantized_matmul 接入 |
| P2-1 | `tests/test_backend_compat.rs` | 新增 |
| P2-2 | `tests/test_scheduler_refactor.rs` | 新增 |
| P2-3 | `tests/e2e_alignment/` | 新增目录 |
