# 算法路径覆盖矩阵 (Algorithm Path Coverage)

> 按算法路径去重的模型测试覆盖分析。同一算法路径只需一个最小代表模型做 E2E 测试。

## 1. 算法路径定义

一条"算法路径"由以下 5 个维度唯一确定：

| 维度 | 说明 |
|------|------|
| **FFN** | SwiGLU / GeGLU / Standard GELU / MoE |
| **注意力** | Standard Causal / Sliding Window / Softcap / Bidirectional |
| **位置编码** | Full RoPE / Partial RoPE / Absolute / Absolute+TokenType |
| **归一化** | RMSNorm / LayerNorm+Bias |
| **特殊算子** | LogitSoftCap / MoERouter+Dispatch / 无 |

**不构成算法差异的因素**（仅权重布局差异）：
- Fused QKV (glm4) vs Separate QKV (qwen3) → Split 是元数据 op，数学等价
- Fused gate_up_proj (phi4) vs Separate gate/up (qwen3) → 同上
- GQA vs MHA → 仅 `num_kv_heads` 参数不同，代码路径一致

## 2. Generator 算法路径

### 8 个模板 → 7 条唯一路径

| 路径 ID | FFN | 注意力 | 位置编码 | 归一化 | 特殊算子 | 模板 |
|---------|-----|--------|---------|--------|---------|------|
| **G-A** | SwiGLU | Standard | Full RoPE | RMSNorm | — | qwen3, llama, glm4 |
| **G-B** | SwiGLU | **Sliding Window** | Full RoPE | RMSNorm | — | mistral3 |
| **G-C** | SwiGLU | Standard | **Partial RoPE** | RMSNorm | — | phi4 |
| **G-D** | **Std GELU** (non-gated) | **Sliding+Global 交替** | **Dual RoPE (sliding θ=10K / global θ=1M+p-RoPE partial=0.25)** | RMSNorm + **QkNorm** + **ValueNorm** | **Per-Layer Embeddings (PLE)** | gemma4 |
| **G-E** | **Std GELU** | Standard | **Absolute** | **LayerNorm+Bias** | Bias throughout | gpt2next |
| **G-F** | SwiGLU+**MoE** | Standard | Full RoPE | RMSNorm | **Router+SharedExperts+Dispatch** | deepseek |
| **G-G** | SwiGLU+**MoE** | Standard | Full RoPE | RMSNorm | **QwenRouter** | qwen3 (MoE variant) |

### 模型→路径映射

| 路径 | 所有模型 | E2E 代表 | 状态 |
|------|---------|---------|------|
| G-A | Qwen3, Qwen2.5, Llama4, SmolLM2, SmolLM3, InternLM3, GLM4, GLM5 | **SmolLM2-135M** (ST) + **Qwen3-0.6B** (GGUF) | ✅ 已覆盖 |
| G-B | Mistral3, Ministral | — | ❌ 未覆盖 |
| G-C | Phi4, Phi4-mini | — | ❌ 未覆盖 |
| G-D | Gemma4-E2B/E4B/31B/26B-A4B | `google/gemma-4-E2B` (模拟加载 + dry-run) | 🟢 atomic 契约 (T36) + pattern_fusion (T41/T42) + JIT compile (T30/T37/T38) 全过,待 E2E 数值对齐 (T47) |
| G-E | GPT-OSS-1.5B, GPT-OSS-12B | — | ❌ 未覆盖 |
| G-F | DeepSeek | — | ❌ 未覆盖 (模型过大) |
| G-G | Qwen3MoE (235B), Llama4-Scout, Llama4-8B | — | ❌ 未覆盖 (模型过大) |

## 3. Encoder 算法路径

### 1 条唯一路径

| 路径 ID | FFN | 注意力 | 位置编码 | 归一化 | 特殊算子 | 模板 |
|---------|-----|--------|---------|--------|---------|------|
| **E-A** | Std GELU | **Bidirectional** | **Absolute+TokenType** | **LayerNorm+Bias** | Bias throughout | xlmr |

### 模型→路径映射

| 路径 | 所有模型 | E2E 代表 | 状态 |
|------|---------|---------|------|
| E-A | XlmR: e5-small/base/large, bge-m3, m3e-base, jina-v2/v4; XlmRNext: bge-m4, bge-reranker-v3 | **e5-small-v2** (ST) + **multilingual-e5-small** (ONNX) + **bge-reranker-v2-m3** (ST/ONNX) + **Qwen3-Embedding** (GGUF) | ✅ 已覆盖 |

> XlmR 和 XlmRNext 共用 xlmr.yaml 模板，算法路径完全一致。

## 4. 缺口分析

### 4.1 可填补的缺口 (有公开可用的小模型)

| 路径 | 最小可用模型 | 大小 | 可行性 |
|------|------------|------|--------|
| **G-D** (Gemma4 Dual RoPE + QkNorm) | `google/gemma-4-E2B` | Effective 2B | ✅ 可行，模型公开可用 |
| **G-E** (Legacy GELU) | `openai/gpt-oss-1.5b` | 1.5B | ✅ 已注册，模板通过 build.rs 自动扫描 |

### 4.2 需要 GGUF 量化版的缺口 (原始模型过大)

| 路径 | 最小原始模型 | 大小 | 备选 GGUF |
|------|------------|------|----------|
| **G-B** (Sliding Window) | `mistralai/Mistral-Small-3.2` | 3.2B | 寻找 Q4 量化版 |
| **G-C** (Partial RoPE) | `microsoft/Phi-4-mini-instruct` | 3.8B | 寻找 Q4 量化版 |

### 4.3 无法 E2E 的缺口 (模型太大，仅单元测试覆盖)

| 路径 | 最小模型 | 大小 | 覆盖策略 |
|------|---------|------|---------|
| **G-F** (DeepSeek MoE) | DeepSeek-V3 | 671B | 单元测试覆盖 MoERouter+MoEDispatch |
| **G-G** (Qwen MoE) | Qwen3-MoE-A22B | 235B | 单元测试覆盖 MoE 路由 |

### 4.4 代码缺口

| 问题 | 详情 | 状态 |
|------|------|------|
| ~~GPT-OSS 未注册~~ | `gpt2next.yaml` 模板已通过 build.rs 扫描自动注册 | ✅ 已修复 |

## 5. E2E 测试最终方案 (每路径一个代表)

### Generator

| 路径 | 代表模型 | 格式 | 测试内容 |
|------|---------|------|---------|
| G-A | SmolLM2-135M / Qwen3-0.6B | ST+GGUF+ONNX | ✅ 已有 |
| G-D | gemma-4-E2B | ST (或 GGUF) | 🔴 需新增：QkNorm + ValueNorm + DualRoPE (sliding+global p-RoPE) + PLE + Sliding-window/Global 交替 attention |
| G-E | openai/gpt-oss-1.5b | ST | 🔴 需新增：LayerNorm+Bias + GELU + AbsolutePos + FusedQKV |
| G-B | mistral-small-3.2 Q4_K_M | GGUF | 🟡 可选：Sliding Window |
| G-C | phi-4-mini Q4_K_M | GGUF | 🟡 可选：Partial RoPE |

### Encoder (Embedding / Reranker)

| 路径 | 代表模型 | 格式 | 测试内容 |
|------|---------|------|---------|
| E-A | e5-small-v2 / bge-reranker-v2-m3 | ST+GGUF+ONNX | ✅ 已有 |

### 融合管线

| 测试 | 模型组合 | 测试内容 |
|------|---------|---------|
| Embed+Rerank | e5-small + bge-reranker | ✅ 已有 |
| RAG | e5-small + bge-reranker + SmolLM2-135M | ✅ 已有 |
