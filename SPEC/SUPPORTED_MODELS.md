# Supported Models (SSOT) - 2026 SOTA Edition

> **📌 SSOT**: 本文档是 `gllm` 支持的所有模型的唯一真源。
> **Constraint**: 严格遵循 **"Latest Version Only"** 策略。
> **Strategy**: 同一系列出新版后，立即废弃旧版（e.g., Qwen3 发布即删除 Qwen2.5）。
> **Exception (REQ-MODEL-LATEST)**: 仅允许在本表中**显式标注**的 Legacy Exception。

## 1. Generator Models (文本生成)

### Qwen Series (Alibaba)
> Status: **Qwen3 (Jan 2026)** is the current standard. Qwen2.5 is deprecated.

| Model ID | Architecture | Specs | HF Repo | ModelScope Repo | 特性 |
|----------|--------------|-------|---------|-----------------|------|
| `qwen3-7b` | Qwen3 | 7B Dense | `Qwen/Qwen3-7B-Instruct` | `qwen/Qwen3-7B-Instruct` | 通用标杆 |
| `qwen3-moe` | Qwen3MoE | A22B (235B) | `Qwen/Qwen3-235B-A22B-Instruct` | `qwen/Qwen3-235B-A22B-Instruct` | 旗舰 MoE, 双模式(Thinking) |
| `qwen3-thinking`| Qwen3 | 32B | `Qwen/Qwen3-Max-Thinking` | `qwen/Qwen3-Max-Thinking` | 强化推理能力 |

### Llama Series (Meta)
> Status: **Llama 4 (Apr 2025)** is the current standard. Llama 3.x is deprecated.

| Model ID | Architecture | Specs | HF Repo | ModelScope Repo | 特性 |
|----------|--------------|-------|---------|-----------------|------|
| `llama-4-8b` | Llama4 | 8B MoE | `meta-llama/Llama-4-8B-Instruct` | `LLM-Research/Llama-4-8B-Instruct` | 原生多模态/多语言 |
| `llama-4-scout`| Llama4 | 17B (Active) | `meta-llama/Llama-4-Scout` | `LLM-Research/Llama-4-Scout` | 16 Experts MoE |

### Mistral Series (Mistral AI)
> Status: **Mistral 3 / Ministral (Dec 2025)** is the current standard.

| Model ID | Architecture | Specs | HF Repo | ModelScope Repo | 特性 |
|----------|--------------|-------|---------|-----------------|------|
| `ministral-8b` | Ministral | 8B | `mistralai/Ministral-8B-Instruct` | `AI-ModelScope/Ministral-8B-Instruct` | 边缘侧 SOTA |
| `mistral-small-3`| Mistral3 | 3.2 (Update) | `mistralai/Mistral-Small-3.2` | `AI-ModelScope/Mistral-Small-3.2` | 高效推理 |

### GLM Series (Zhipu AI)
> Status: **GLM-4.7 (Dec 2025)** is the current standard.

| Model ID | Architecture | Specs | HF Repo | ModelScope Repo | 特性 |
|----------|--------------|-------|---------|-----------------|------|
| `glm-4.7-flash`| GLM-4 | 3B (Active) | `THUDM/glm-4.7-flash` | `ZhipuAI/glm-4.7-flash` | MoE 架构, 免费级首选 |
| `glm-5-9b` | GLM-5 | 9B | `THUDM/glm-5-9b-chat` | `ZhipuAI/glm-5-9b-chat` | 下一代架构 (Preview) |

### GPT-OSS Series (OpenAI)
> Status: **GPT-OSS (2026)** open weights release.

| Model ID | Architecture | Specs | HF Repo | ModelScope Repo | 特性 |
|----------|--------------|-------|---------|-----------------|------|
| `gpt-oss-1.5b` | GPT-2-Next | 1.5B | `openai/gpt-oss-1.5b` | `openai-mirror/gpt-oss-1.5b` | Fused QKV 经典架构回归 |
| `gpt-oss-12b` | GPT-2-Next | 12B | `openai/gpt-oss-12b` | `openai-mirror/gpt-oss-12b` | 强推理能力 |

### Phi Series (Microsoft)
> Status: **Phi-4 (2025)** 轻量主力，**Phi-4-mini (Feb 2025)** SLM 标准。**Legacy Exception**: 低资源设备与兼容性保留（早于 2025-09）。

| Model ID | Architecture | Specs | HF Repo | ModelScope Repo | 特性 |
|----------|--------------|-------|---------|-----------------|------|
| `phi-4` | Phi4 | 14B Dense | `microsoft/Phi-4` | - | Phi 系列主力版本 |
| `phi-4-mini` | Phi4 | 3.8B Dense | `microsoft/Phi-4-mini-instruct` | - | 端侧小模型/低延迟推理 |

### SmolLM Series (HuggingFaceTB)
> Status: **SmolLM (2025)** 轻量级高效模型系列。

| Model ID | Architecture | Specs | HF Repo | ModelScope Repo | 特性 |
|----------|--------------|-------|---------|-----------------|------|
| `smollm2-135m` | Llama4 | 135M | `HuggingFaceTB/SmolLM2-135M-Instruct` | - | 超轻量端侧模型 |
| `smollm3-3b` | Llama4 | 3B | `HuggingFaceTB/SmolLM3-3B` | - | 轻量级多用途 |

### InternLM Series (Shanghai AI Lab)
> Status: **Internlm3 (2025)** 开源高效模型。

| Model ID | Architecture | Specs | HF Repo | ModelScope Repo | 特性 |
|----------|--------------|-------|---------|-----------------|------|
| `internlm3-8b` | Llama4 | 8B | `internlm/internlm3-8b-instruct` | - | 高效推理 |

### Gemma Series (Google)
> Status: **Gemma 2 (Jun/Jul 2024)** 系列。**Legacy Exception**: 轻量对话兼容性保留（早于 2025-09）。

| Model ID | Architecture | Specs | HF Repo | ModelScope Repo | 特性 |
|----------|--------------|-------|---------|-----------------|------|
| `gemma-2-2b-it` | Gemma2 | 2B | `google/gemma-2-2b-it` | - | 轻量对话模型 |
| `gemma-2-9b` | Gemma2 | 9B | `google/gemma-2-9b-it` | - | 中型对话模型 |
| `gemma-2-27b` | Gemma2 | 27B | `google/gemma-2-27b-it` | - | 大模型对话能力 |

## 2. Embedding Models (文本向量化)

### Qwen Series (Alibaba)

| Model ID | Architecture | Dims | HF Repo | ModelScope Repo | 说明 |
|----------|--------------|------|---------|-----------------|------|
| `qwen3-embed` | Qwen3 | 2048 | `Qwen/Qwen3-Embedding` | `qwen/Qwen3-Embedding` | Qwen3 原生向量 |

### BGE Series (BAAI)

| Model ID | Architecture | Dims | HF Repo | ModelScope Repo | 说明 |
|----------|--------------|------|---------|-----------------|------|
| `bge-m3` | XLM-R | 1024 | `BAAI/bge-m3` | - | **Legacy Exception**: M4 已有但 M3 仍广泛使用 |
| `bge-m4` | XLM-R-Next | 1536 | `BAAI/bge-m4` | `Xorbits/bge-m4` | M3 升级版 |

### E5 Series (intfloat)

| Model ID | Architecture | Dims | HF Repo | ModelScope Repo | 说明 |
|----------|--------------|------|---------|-----------------|------|
| `e5-small` | XlmR | 384 | `intfloat/e5-small` | - | 轻量级嵌入 |
| `e5-base` | XlmR | 768 | `intfloat/e5-base` | - | 标准嵌入 |
| `e5-large` | XlmR | 1024 | `intfloat/e5-large` | - | 高精度嵌入 |

### M3E Series (moka-ai)

| Model ID | Architecture | Dims | HF Repo | ModelScope Repo | 说明 |
|----------|--------------|------|---------|-----------------|------|
| `m3e-base` | XlmR | 768 | `moka-ai/m3e-base` | - | 中文通用嵌入 |

### Jina Embeddings (jinaai)

| Model ID | Architecture | Dims | HF Repo | ModelScope Repo | 说明 |
|----------|--------------|------|---------|-----------------|------|
| `jina-embeddings-v2-base` | XlmR | 768 | `jinaai/jina-embeddings-v2-base-en` | - | 英文通用嵌入 |
| `jina-embeddings-v2-small` | XlmR | 384 | `jinaai/jina-embeddings-v2-small-en` | - | 轻量英文嵌入 |
| `jina-embeddings-v4` | XlmR | 768 | `jinaai/jina-embeddings-v4` | - | 最新通用嵌入 |

## 3. Rerank Models (文本重排序)

| Model ID | Architecture | HF Repo | ModelScope Repo | 说明 |
|----------|--------------|---------|-----------------|------|
| `qwen3-rerank` | Qwen3 | `Qwen/Qwen3-Reranker` | `qwen/Qwen3-Reranker` | 原生重排序 |
| `bge-reranker-v2-m3` | XlmR | `BAAI/bge-reranker-v2-m3` | - | 轻量级中文重排序 |
| `bge-rerank-v3` | XLM-R-Next | `BAAI/bge-reranker-v3` | `Xorbits/bge-reranker-v3` | V2 升级版 |

## 4. Weight Loading Specs (Next-Gen)

针对 2026 年新架构的适配规范：

- **Llama 4 (MoE)**:
  - 必须支持 `experts` 路由权重解析。
  - 必须处理 Image/Text 多模态输入的 Projector 权重。
- **Qwen3 (Thinking)**:
  - 支持 `thinking_head` 或特殊的 System Prompt 触发权重。
  - **Qwen3-ASR/TTS**: 暂不支持音频组件，仅支持 LLM/Embedding 部分。
