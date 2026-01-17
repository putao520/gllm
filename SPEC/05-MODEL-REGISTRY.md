# gllm 模型注册表

## 概述

本文档定义 gllm 支持的所有模型类型、架构分支、量化格式的完整注册表。作为 SSOT（Single Source of Truth），所有模型相关的实现和测试必须以本文档为准。

**实现位置**: `src/registry.rs`

---

## 模型类型 (ModelType)

| 类型 | 描述 | 用途 |
|------|------|------|
| `Embedding` | 嵌入编码模型 | 文本向量化、语义搜索 |
| `Rerank` | 重排序模型 | 文档相关性排序 |
| `Generator` | 生成解码模型 | 文本生成、对话 |

---

## 架构分支 (Architecture)

### Embedding 架构

| 架构 | 描述 | 代表模型 |
|------|------|----------|
| `Bert` | BERT 编码器 | BGE, all-MiniLM, E5 |
| `Qwen2Embedding` | Qwen2 嵌入变体 | SFR-Embedding-Code-2B |
| `Qwen3Embedding` | Qwen3 嵌入变体 | Qwen3-Embedding-0.6B/4B/8B |
| `MistralEmbedding` | Mistral 嵌入变体 | SFR-Embedding-Code-7B |
| `JinaV4` | Jina v4 架构 | jina-embeddings-v4 |
| `NVIDIANemotron` | NVIDIA Nemotron | llama-embed-nemotron-8b |

### Rerank 架构

| 架构 | 描述 | 代表模型 |
|------|------|----------|
| `CrossEncoder` | 交叉编码器 | BGE-Reranker, MS-MARCO |
| `Qwen3Reranker` | Qwen3 重排序变体 | Qwen3-Reranker-0.6B/4B/8B |
| `JinaRerankerV3` | Jina Reranker v3 | jina-reranker-v3 |

### Generator 架构

| 架构 | 描述 | 代表模型 | 参数规模 |
|------|------|----------|----------|
| `Qwen2Generator` | Qwen2/2.5 解码器 | Qwen2.5-0.5B~72B | 0.5B~72B |
| `Qwen3Generator` | Qwen3 解码器 | Qwen3-0.6B~32B | 0.6B~32B |
| `Qwen3MoE` | Qwen3 MoE 架构 | Qwen3-30B-A3B, Qwen3-235B-A22B | 30B/235B |
| `MistralGenerator` | Mistral 解码器 | Mistral-7B-Instruct | 7B |
| `Mixtral` | Mixtral MoE 架构 | Mixtral-8x7B, 8x22B | 47B/141B |
| `Phi3Generator` | Phi-4 解码器 | Phi-4, Phi-4-mini | ~3B |
| `SmolLM3Generator` | SmolLM3 解码器 | SmolLM3-3B | 3B |
| `InternLM3Generator` | InternLM3 解码器 | InternLM3-8B | 8B |
| `GLM4` | GLM-4 解码器 | GLM-4-9B-Chat | 9B |
| `GLM4MoE` | GLM-4.7 MoE 架构 | GLM-4.7 | MoE |
| `DeepSeekV3` | DeepSeek-V3 MoE | DeepSeek-V3 | 671B |
| `GptOss` | OpenAI GPT-OSS MoE | gpt-oss-20b/120b | 20B/120B |
| `Gemma3n` | Google Gemma 3n | Gemma-3n | - |

---

## 量化格式 (Quantization)

| 格式 | 别名后缀 | Repo 后缀 | 描述 |
|------|----------|-----------|------|
| `None` | (无) | (无) | 全精度 (FP16/BF16) |
| `Int4` | `:int4`, `:4bit` | `-Int4` | 4-bit 整数量化 |
| `Int8` | `:int8`, `:8bit` | `-Int8` | 8-bit 整数量化 |
| `AWQ` | `:awq` | `-AWQ` | AWQ 量化 |
| `GPTQ` | `:gptq` | `-GPTQ` | GPTQ 量化 |
| `GGUF` | `:gguf` | `-GGUF` | GGUF 格式 (llama.cpp 兼容) |
| `BNB4` | `:bnb4`, `:bnb-4bit` | `-bnb-4bit` | bitsandbytes 4-bit |
| `BNB8` | `:bnb8`, `:bnb-8bit` | `-bnb-8bit` | bitsandbytes 8-bit |
| `FP8` | `:fp8` | `-FP8` | FP8 量化 |

---

## Embedding 模型注册表

### BGE 系列

| 别名 | HuggingFace Repo | 架构 | 维度 | 参数 |
|------|------------------|------|------|------|
| `bge-small-zh` | BAAI/bge-small-zh-v1.5 | Bert | 512 | 24M |
| `bge-small-en` | BAAI/bge-small-en-v1.5 | Bert | 384 | 33M |
| `bge-base-en` | BAAI/bge-base-en-v1.5 | Bert | 768 | 109M |
| `bge-large-en` | BAAI/bge-large-en-v1.5 | Bert | 1024 | 335M |

### Sentence Transformers 系列

| 别名 | HuggingFace Repo | 架构 | 维度 | 参数 |
|------|------------------|------|------|------|
| `all-MiniLM-L6-v2` | sentence-transformers/all-MiniLM-L6-v2 | Bert | 384 | 22M |
| `all-MiniLM-L12-v2` | sentence-transformers/all-MiniLM-L12-v2 | Bert | 384 | 33M |
| `all-mpnet-base-v2` | sentence-transformers/all-mpnet-base-v2 | Bert | 768 | 109M |
| `paraphrase-MiniLM-L6-v2` | sentence-transformers/paraphrase-MiniLM-L6-v2 | Bert | 384 | 22M |
| `multi-qa-mpnet-base-dot-v1` | sentence-transformers/multi-qa-mpnet-base-dot-v1 | Bert | 768 | 109M |
| `all-distilroberta-v1` | sentence-transformers/all-distilroberta-v1 | Bert | 768 | 82M |

### E5 系列

| 别名 | HuggingFace Repo | 架构 | 维度 | 参数 |
|------|------------------|------|------|------|
| `e5-small` | intfloat/e5-small | Bert | 384 | 33M |
| `e5-base` | intfloat/e5-base | Bert | 768 | 109M |
| `e5-large` | intfloat/e5-large | Bert | 1024 | 335M |

### Jina 系列

| 别名 | HuggingFace Repo | 架构 | 维度 |
|------|------------------|------|------|
| `jina-embeddings-v2-small-en` | jinaai/jina-embeddings-v2-small-en | Bert | 512 |
| `jina-embeddings-v2-base-en` | jinaai/jina-embeddings-v2-base-en | Bert | 768 |
| `jina-embeddings-v4` | jinaai/jina-embeddings-v4 | JinaV4 | - |

### 中文/多语言模型

| 别名 | HuggingFace Repo | 架构 | 维度 |
|------|------------------|------|------|
| `m3e-base` | moka-ai/m3e-base | Bert | 768 |
| `multilingual-MiniLM-L12-v2` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Bert | 384 |
| `distiluse-base-multilingual-cased-v1` | sentence-transformers/distiluse-base-multilingual-cased-v1 | Bert | 512 |

### 代码嵌入模型

| 别名 | HuggingFace Repo | 架构 | 参数 |
|------|------------------|------|------|
| `codebert-base` | claudios/codebert-base | Bert | 125M |
| `graphcodebert-base` | claudios/graphcodebert-base | Bert | 125M |
| `unixcoder-base` | claudios/unixcoder-base | Bert | 125M |
| `starencoder` | bigcode/starencoder | Bert | - |
| `codexembed-400m` | Salesforce/SFR-Embedding-Code-400M_R | Bert | 400M |
| `codexembed-2b` | Salesforce/SFR-Embedding-Code-2B_R | Qwen2Embedding | 2B |
| `codexembed-7b` | Salesforce/SFR-Embedding-Code-7B_R | MistralEmbedding | 7B |

### Qwen3 Embedding 系列

| 别名 | HuggingFace Repo | 架构 | 参数 |
|------|------------------|------|------|
| `qwen3-embedding-0.6b` | Qwen/Qwen3-Embedding-0.6B | Qwen3Embedding | 0.6B |
| `qwen3-embedding-4b` | Qwen/Qwen3-Embedding-4B | Qwen3Embedding | 4B |
| `qwen3-embedding-8b` | Qwen/Qwen3-Embedding-8B | Qwen3Embedding | 8B |

### 其他 Embedding 模型

| 别名 | HuggingFace Repo | 架构 |
|------|------------------|------|
| `llama-embed-nemotron-8b` | nvidia/llama-embed-nemotron-8b | NVIDIANemotron |

---

## Rerank 模型注册表

### BGE Reranker 系列

| 别名 | HuggingFace Repo | 架构 | 参数 |
|------|------------------|------|------|
| `bge-reranker-base` | BAAI/bge-reranker-base | CrossEncoder | 278M |
| `bge-reranker-large` | BAAI/bge-reranker-large | CrossEncoder | 560M |
| `bge-reranker-v2` | BAAI/bge-reranker-v2-m3 | CrossEncoder | 568M |

### MS-MARCO Reranker 系列

| 别名 | HuggingFace Repo | 架构 | 参数 |
|------|------------------|------|------|
| `ms-marco-TinyBERT-L-2-v2` | cross-encoder/ms-marco-TinyBERT-L-2-v2 | CrossEncoder | 4.4M |
| `ms-marco-MiniLM-L-6-v2` | cross-encoder/ms-marco-MiniLM-L-6-v2 | CrossEncoder | 22M |
| `ms-marco-MiniLM-L-12-v2` | cross-encoder/ms-marco-MiniLM-L-12-v2 | CrossEncoder | 33M |
| `ms-marco-electra-base` | cross-encoder/ms-marco-electra-base | CrossEncoder | 109M |

### Qwen3 Reranker 系列

| 别名 | HuggingFace Repo | 架构 | 参数 |
|------|------------------|------|------|
| `qwen3-reranker-0.6b` | Qwen/Qwen3-Reranker-0.6B | Qwen3Reranker | 0.6B |
| `qwen3-reranker-4b` | Qwen/Qwen3-Reranker-4B | Qwen3Reranker | 4B |
| `qwen3-reranker-8b` | Qwen/Qwen3-Reranker-8B | Qwen3Reranker | 8B |

### 其他 Reranker 模型

| 别名 | HuggingFace Repo | 架构 |
|------|------------------|------|
| `quora-distilroberta-base` | cross-encoder/quora-distilroberta-base | CrossEncoder |
| `jina-reranker-v3` | jinaai/jina-reranker-v3 | JinaRerankerV3 |

---

## Generator 模型注册表

### Qwen2/2.5 系列

| 别名 | HuggingFace Repo | 架构 | 参数 |
|------|------------------|------|------|
| `qwen2-7b-instruct` | Qwen/Qwen2-7B-Instruct | Qwen2Generator | 7B |
| `qwen2.5-0.5b-instruct` | Qwen/Qwen2.5-0.5B-Instruct | Qwen2Generator | 0.5B |
| `qwen2.5-1.5b-instruct` | Qwen/Qwen2.5-1.5B-Instruct | Qwen2Generator | 1.5B |
| `qwen2.5-3b-instruct` | Qwen/Qwen2.5-3B-Instruct | Qwen2Generator | 3B |
| `qwen2.5-7b-instruct` | Qwen/Qwen2.5-7B-Instruct | Qwen2Generator | 7B |
| `qwen2.5-14b-instruct` | Qwen/Qwen2.5-14B-Instruct | Qwen2Generator | 14B |
| `qwen2.5-32b-instruct` | Qwen/Qwen2.5-32B-Instruct | Qwen2Generator | 32B |
| `qwen2.5-72b-instruct` | Qwen/Qwen2.5-72B-Instruct | Qwen2Generator | 72B |

### Qwen3 系列

| 别名 | HuggingFace Repo | 架构 | 参数 |
|------|------------------|------|------|
| `qwen3-0.6b` | Qwen/Qwen3-0.6B | Qwen3Generator | 0.6B |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | Qwen3Generator | 1.7B |
| `qwen3-4b` | Qwen/Qwen3-4B | Qwen3Generator | 4B |
| `qwen3-8b` | Qwen/Qwen3-8B | Qwen3Generator | 8B |
| `qwen3-14b` | Qwen/Qwen3-14B | Qwen3Generator | 14B |
| `qwen3-32b` | Qwen/Qwen3-32B | Qwen3Generator | 32B |

### Qwen3 MoE 系列

| 别名 | HuggingFace Repo | 架构 | 参数 |
|------|------------------|------|------|
| `qwen3-30b-a3b` | Qwen/Qwen3-30B-A3B | Qwen3MoE | 30B (激活 3B) |
| `qwen3-235b-a22b` | Qwen/Qwen3-235B-A22B | Qwen3MoE | 235B (激活 22B) |

### Mistral/Mixtral 系列

| 别名 | HuggingFace Repo | 架构 | 参数 | GGUF 源 |
|------|------------------|------|------|---------|
| `mistral-7b-instruct` | mistralai/Mistral-7B-Instruct-v0.2 | MistralGenerator | 7B | bartowski |
| `mixtral-8x7b-instruct` | mistralai/Mixtral-8x7B-Instruct-v0.1 | Mixtral | 47B | bartowski |
| `mixtral-8x22b-instruct` | mistralai/Mixtral-8x22B-Instruct-v0.1 | Mixtral | 141B | bartowski |

### Phi 系列

| 别名 | HuggingFace Repo | 架构 | 参数 | GGUF 源 |
|------|------------------|------|------|---------|
| `phi-4` | microsoft/phi-4 | Phi3Generator | ~14B | - |
| `phi-4-mini-instruct` | microsoft/phi-4-mini-instruct | Phi3Generator | ~3B | bartowski |

### GLM 系列

| 别名 | HuggingFace Repo | 架构 | 参数 | GGUF 源 |
|------|------------------|------|------|---------|
| `glm-4-9b-chat` | THUDM/glm-4-9b-chat-hf | GLM4 | 9B | bartowski |
| `glm-4.7` | zai-org/GLM-4.7 | GLM4MoE | MoE | - |

### 其他 Generator 模型

| 别名 | HuggingFace Repo | 架构 | 参数 | GGUF 源 |
|------|------------------|------|------|---------|
| `smollm3-3b` | HuggingFaceTB/SmolLM3-3B | SmolLM3Generator | 3B | bartowski |
| `internlm3-8b-instruct` | internlm/internlm3-8b-instruct | InternLM3Generator | 8B | bartowski |
| `deepseek-v3` | deepseek-ai/DeepSeek-V3 | DeepSeekV3 | 671B | - |
| `gpt-oss-20b` | openai/gpt-oss-20b | GptOss | 20B | - |
| `gpt-oss-120b` | openai/gpt-oss-120b | GptOss | 120B | - |

---

## GGUF 第三方源映射

以下模型使用第三方 GGUF 仓库（bartowski 等）：

| 别名 | 官方 Repo | GGUF Repo |
|------|-----------|-----------|
| `mistral-7b-instruct` | mistralai/Mistral-7B-Instruct-v0.2 | bartowski/Mistral-7B-Instruct-v0.2-GGUF |
| `glm-4-9b-chat` | THUDM/glm-4-9b-chat-hf | bartowski/glm-4-9b-chat-GGUF |
| `phi-4-mini-instruct` | microsoft/phi-4-mini-instruct | bartowski/phi-4-mini-instruct-GGUF |
| `smollm3-3b` | HuggingFaceTB/SmolLM3-3B | bartowski/SmolLM3-3B-GGUF |
| `internlm3-8b-instruct` | internlm/internlm3-8b-instruct | bartowski/internlm3-8b-instruct-GGUF |
| `mixtral-8x7b-instruct` | mistralai/Mixtral-8x7B-Instruct-v0.1 | bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF |
| `mixtral-8x22b-instruct` | mistralai/Mixtral-8x22B-Instruct-v0.1 | bartowski/Mixtral-8x22B-Instruct-v0.1-GGUF |

---

## 别名解析规则

### 基本格式

```
{alias}                    → 默认精度 (FP16/BF16)
{alias}:{quantization}     → 指定量化格式
{org}/{model}              → 直接 HuggingFace Repo ID
```

### 示例

```rust
// 默认精度
registry.resolve("qwen2.5-0.5b-instruct")
// → Qwen/Qwen2.5-0.5B-Instruct

// GGUF 量化
registry.resolve("qwen2.5-0.5b-instruct:gguf")
// → Qwen/Qwen2.5-0.5B-Instruct-GGUF

// Int4 量化
registry.resolve("qwen3-embedding-0.6b:int4")
// → Qwen/Qwen3-Embedding-0.6B-Int4

// 第三方 GGUF
registry.resolve("glm-4-9b-chat:gguf")
// → bartowski/glm-4-9b-chat-GGUF

// 直接 Repo ID
registry.resolve("Qwen/Qwen2.5-7B-Instruct-AWQ")
// → Qwen/Qwen2.5-7B-Instruct-AWQ (自动推断 AWQ 量化)
```

---

## 统计概览

| 类别 | 数量 |
|------|------|
| 总注册模型 | ~70+ |
| Embedding 模型 | ~30 |
| Rerank 模型 | ~14 |
| Generator 模型 | ~30 |
| 支持的架构 | 22 |
| 量化格式 | 9 |

---

## 验证测试结果 (2025-01-17)

### Embedding 模型测试

| 模型 | 状态 | 备注 |
|------|------|------|
| bge-small-en | ✅ 通过 | 代表性 BERT 模型 |
| all-MiniLM-L6-v2 | ✅ 通过 | Sentence Transformers |
| e5-small | ✅ 通过 | E5 系列 |

### Rerank 模型测试

| 模型 | 状态 | 备注 |
|------|------|------|
| bge-reranker-base | ✅ 通过 | BGE Reranker |
| ms-marco-MiniLM-L-6-v2 | ✅ 通过 | MS-MARCO |

### Generator 架构测试

| 架构 | 模型 | FP16 | GGUF | 备注 |
|------|------|------|------|------|
| Qwen2Generator | qwen2.5-0.5b-instruct | ✅ | ✅ | |
| Qwen3Generator | qwen3-0.6b | ✅ | ✅ | |
| MistralGenerator | mistral-7b-instruct | ⏭️ 跳过 | ✅ | 7B 超出 6GB VRAM |
| Phi3Generator | phi-4-mini-instruct | ⏭️ 跳过 | ✅ | ~3B 超出 6GB VRAM |
| SmolLM3Generator | smollm3-3b | ⏭️ 跳过 | ✅ | 3B 超出 6GB VRAM |
| InternLM3Generator | internlm3-8b-instruct | ⏭️ 跳过 | ⚠️ 失败 | 需要 `quantized` feature |
| GLM4 | glm-4-9b-chat | ⏭️ 跳过 | ✅ | 9B 超出 6GB VRAM |
| Qwen3MoE | qwen3-30b-a3b | ⏭️ 跳过 | ⏭️ 跳过 | 无 GGUF repo |

### 测试统计

```
FP16:  2 passed, 0 failed, 6 skipped
GGUF:  6 passed, 1 failed, 1 skipped
Total: 8 passed, 1 failed
```

**已知问题**:
- InternLM3 GGUF 需要启用 `quantized` feature flag
