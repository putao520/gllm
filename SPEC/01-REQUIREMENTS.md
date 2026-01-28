# gllm 功能需求清单

> **📌 SSOT**: 本文档是 gllm 项目的功能需求唯一真源。

## 1. 模型支持 (REQ-MODEL)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-MODEL-001** | Embedding 模型支持 | 支持 BERT, RoBERTa, JINA, CodeXEmbed 架构 | E2E 测试通过 | ✅ 已完成 |
| **REQ-MODEL-002** | Rerank 模型支持 | 支持 Cross-Encoder, Qwen3-Reranker | E2E 测试通过 | ✅ 已完成 |
| **REQ-MODEL-003** | Generator 模型支持 | 支持 Qwen3, Mistral, GLM-4, Phi-4 | E2E 测试通过 | ✅ 已完成 |
| **REQ-MODEL-004** | MoE 模型支持 | 支持 Qwen3-MoE, GLM-4.7, Mixtral | E2E 测试通过 | ✅ 已完成 |
| **REQ-MODEL-005** | GPT-OSS 模型支持 | 支持 OpenAI GPT-OSS (Fused QKV) 架构 | 1. `registry` 解析正确<br>2. `loader` 正确处理 `c_attn`/`c_proj` 权重<br>3. 能够成功加载 GPT-OSS 权重 | ✅ 已完成 |

## 2. 核心功能 (REQ-CORE)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-CORE-001** | 自动后端检测 | 自动选择 CUDA/ROCm/Metal/CPU | `detect_backend()` 返回正确类型 | ✅ 已完成 |
| **REQ-CORE-002** | 自动降级 | GPU OOM 时自动降级到 CPU | `FallbackEmbedder` 正常工作 | ✅ 已完成 |
| **REQ-CORE-003** | 量化支持 | 支持 Int4/Int8/AWQ/GPTQ/GGUF 加载 | 能够加载并推理量化模型 | ✅ 已完成 |

## 3. 架构约束 (REQ-ARCH)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-ARCH-001** | 零拷贝推理 | 推理过程中 GPU 数据不回传 CPU | 符合 ARCH-GPU-001 | ✅ 已完成 |
| **REQ-ARCH-002** | 单一后端原则 | 全程在单一后端执行 | 符合 ARCH-SINGLE-BACKEND | ✅ 已完成 |
