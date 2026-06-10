# 支持模型清单

> **SSOT 声明**: 本文档是 gllm 支持的所有模型的唯一真源。
> **版本策略 (REQ-MODEL-LATEST)**: 仅支持 2025年9月以后发布的模型。系列出新版后立即废弃旧版。
> **Legacy Exception**: 仅允许在本表中显式标注的条目。

## 1. Generator Models (文本生成)

### Qwen Series (Alibaba)

| Model ID | Architecture | Specs | HF Repo | 特性 |
|----------|--------------|-------|---------|------|
| `qwen3-7b` | Qwen3 | 7B Dense | `Qwen/Qwen3-7B-Instruct` | 通用标杆 |
| `qwen3-moe` | Qwen3MoE | A22B (235B) | `Qwen/Qwen3-235B-A22B-Instruct` | 旗舰 MoE，双模式 Thinking |
| `qwen3-thinking` | Qwen3 | 32B | `Qwen/Qwen3-Max-Thinking` | 强化推理能力 |

### Llama Series (Meta)

| Model ID | Architecture | Specs | HF Repo | 特性 |
|----------|--------------|-------|---------|------|
| `llama-4-8b` | Llama4 | 8B MoE | `meta-llama/Llama-4-8B-Instruct` | 原生多模态/多语言 |
| `llama-4-scout` | Llama4 | 17B (Active) | `meta-llama/Llama-4-Scout` | 16 Experts MoE |

### Mistral Series (Mistral AI)

| Model ID | Architecture | Specs | HF Repo | 特性 |
|----------|--------------|-------|---------|------|
| `ministral-8b` | Ministral | 8B | `mistralai/Ministral-8B-Instruct` | 边缘侧 SOTA |
| `mistral-small-3` | Mistral3 | 3.2 | `mistralai/Mistral-Small-3.2` | 高效推理 |

### GLM Series (Zhipu AI)

| Model ID | Architecture | Specs | HF Repo | 特性 |
|----------|--------------|-------|---------|------|
| `glm-4.7-flash` | GLM-4 | 3B (Active) | `THUDM/glm-4.7-flash` | MoE 架构 |
| `glm-5-9b` | GLM-5 | 9B | `THUDM/glm-5-9b-chat` | 下一代架构 (Preview) |

### GPT-OSS Series (OpenAI)

| Model ID | Architecture | Specs | HF Repo | 特性 |
|----------|--------------|-------|---------|------|
| `gpt-oss-20b` | GptOss | 20B (MoE, 4 experts/token) | `openai/gpt-oss-20b` | mxfp4 量化 + sliding/full attention 交替 + yarn RoPE |

### Phi Series (Microsoft)

**Legacy Exception**: 低资源设备兼容性保留（早于 2025-09）。

| Model ID | Architecture | Specs | HF Repo | 特性 |
|----------|--------------|-------|---------|------|
| `phi-4` | Phi4 | 14B Dense | `microsoft/Phi-4` | Phi 系列主力 |
| `phi-4-mini` | Phi4 | 3.8B Dense | `microsoft/Phi-4-mini-instruct` | 端侧小模型 |

### SmolLM Series (HuggingFaceTB)

| Model ID | Architecture | Specs | HF Repo | 特性 |
|----------|--------------|-------|---------|------|
| `smollm2-135m` | Llama4 | 135M | `HuggingFaceTB/SmolLM2-135M-Instruct` | 超轻量端侧 |
| `smollm3-3b` | Llama4 | 3B | `HuggingFaceTB/SmolLM3-3B` | 轻量级多用途 |

### InternLM Series (Shanghai AI Lab)

| Model ID | Architecture | Specs | HF Repo | 特性 |
|----------|--------------|-------|---------|------|
| `internlm3-8b` | Llama4 | 8B | `internlm/internlm3-8b-instruct` | 高效推理 |

### DeepSeek Series (DeepSeek)

| Model ID | Architecture | Specs | HF Repo | 特性 |
|----------|--------------|-------|---------|------|
| `deepseek-v3` | DeepSeek | 671B MoE (37B Active) | `deepseek-ai/DeepSeek-V3` | 旗舰 MoE |
| `deepseek-r1` | DeepSeek | 671B MoE | `deepseek-ai/DeepSeek-R1` | 推理强化 |
| `kimi-k2` | DeepSeek | MoE | `moonshotai/Kimi-K2` | DeepSeek 架构变体 |

### Gemma Series (Google)

| Model ID | Architecture | Specs | HF Repo | 特性 |
|----------|--------------|-------|---------|------|
| `gemma-4-e2b` | Gemma4 | Effective 2B (PLE) | `google/gemma-4-E2B` | 轻量对话 / 长上下文 |
| `gemma-4-e4b` | Gemma4 | Effective 4B (PLE) | `google/gemma-4-E4B` | 设备端对话 |
| `gemma-4-31b` | Gemma4 | 31B Dense | `google/gemma-4-31B-IT` | 多模态 (文本+图像+视频帧+音频) |
| `gemma-4-26b-a4b` | Gemma4 | 26B MoE (4B Active) | `google/gemma-4-26B-A4B-it` | MoE 对话 |

**Gemma 4 能力矩阵** (详细状态见 `SUPPORTED_MODELS.md`):

| 能力 | 状态 | 任务 ID |
|------|------|---------|
| CPU JIT codegen (QkNorm/ValueNorm/DualRoPE/PerLayerEmbed/ColumnSlice) | ✅ | T30/T37 |
| GPU codegen (PTX/HIP/MSL) | ✅ | T38 |
| PerLayerEmbedding (PLE,仅 E2B/E4B) | ✅ | T28.2/T28.3/T30/T37 |
| SharedKvRef page 层 | ✅ | T39 (P1.1) |
| SharedKvRef graph 层 | ✅ | T43 (已完成) |
| FusedQkvNormRope 融合 | ✅ | T29/T36/T41/T42 |
| Vision encoder (SigLIP) | ✅ | T44 (已完成,非 stub 测试通过) |
| Audio encoder (USM Conformer) | ✅ | T45 (已完成,非 stub 测试通过) |
| E2E 推理数值验证 | ✅ | dry-run 通过 |

## 2. Embedding Models (文本向量化)

### Qwen Series

| Model ID | Architecture | Dims | HF Repo | 说明 |
|----------|--------------|------|---------|------|
| `qwen3-embed` | Qwen3 | 2048 | `Qwen/Qwen3-Embedding` | Qwen3 原生向量 |

### BGE Series (BAAI)

| Model ID | Architecture | Dims | HF Repo | 说明 |
|----------|--------------|------|---------|------|
| `bge-m3` | XLM-R | 1024 | `BAAI/bge-m3` | **Legacy Exception**: M4 已有但 M3 仍广泛使用 |
| `bge-m4` | XLM-R-Next | 1536 | `BAAI/bge-m4` | M3 升级版 |

### E5 Series (intfloat)

| Model ID | Architecture | Dims | HF Repo | 说明 |
|----------|--------------|------|---------|------|
| `e5-small` | XlmR | 384 | `intfloat/e5-small` | 轻量级嵌入 |
| `e5-base` | XlmR | 768 | `intfloat/e5-base` | 标准嵌入 |
| `e5-large` | XlmR | 1024 | `intfloat/e5-large` | 高精度嵌入 |

### M3E Series (moka-ai)

| Model ID | Architecture | Dims | HF Repo | 说明 |
|----------|--------------|------|---------|------|
| `m3e-base` | XlmR | 768 | `moka-ai/m3e-base` | 中文通用嵌入 |

### Jina Embeddings (jinaai)

| Model ID | Architecture | Dims | HF Repo | 说明 |
|----------|--------------|------|---------|------|
| `jina-embeddings-v2-base` | XlmR | 768 | `jinaai/jina-embeddings-v2-base-en` | 英文通用嵌入 |
| `jina-embeddings-v2-small` | XlmR | 384 | `jinaai/jina-embeddings-v2-small-en` | 轻量英文嵌入 |
| `jina-embeddings-v4` | XlmR | 768 | `jinaai/jina-embeddings-v4` | 最新通用嵌入 |

## 3. Rerank Models (文本重排序)

| Model ID | Architecture | HF Repo | 说明 |
|----------|--------------|---------|------|
| `qwen3-rerank` | Qwen3 | `Qwen/Qwen3-Reranker` | 原生重排序 |
| `bge-reranker-v2-m3` | XlmR | `BAAI/bge-reranker-v2-m3` | 轻量级中文重排序 |
| `bge-rerank-v3` | XLM-R-Next | `BAAI/bge-reranker-v3` | V2 升级版 |

## 4. 格式支持矩阵

| 架构 | SafeTensors | GGUF | ONNX |
|------|------------|------|------|
| Qwen3 / Qwen3MoE | Yes | Yes | Yes |
| Llama4 | Yes | Yes | Yes |
| GLM-4 / GLM-5 | Yes | Yes | Yes |
| Mistral3 / Ministral | Yes | Yes | Yes |
| Phi4 | Yes | Yes | Yes |
| Gemma4 (E2B/E4B/31B/26B-A4B) | Yes | Yes | Yes |
| DeepSeek | Yes | Yes | Yes |
| GptOss (20B MoE) | Yes | — | — |
| XLM-R / XLM-R-Next (Embedding/Rerank) | Yes | — | Yes |
