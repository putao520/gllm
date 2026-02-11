# gllm 项目概述

## 项目目的

gllm 是一个纯 Rust 高性能推理库，用于本地文本嵌入、重排序和文本生成。基于 gllm-kernels 实现 GPU 加速（CUDA/Metal/ROCm），无需外部 SDK。

## 核心特性

- **L3 GPU-Pure 架构** - 零拷贝推理循环（仅 token ID 在 CPU/GPU 间传输）
- **下一代模型** - 原生支持 Qwen3、Llama 4、GLM-5、Mistral 3
- **统一驱动 API** - AOT 编译内核（CUBIN），无运行时编译开销
- **Tree Attention** - 原生支持 speculative decoding
- **多源加载器** - HuggingFace/ModelScope 自动切换
- **量化支持** - Block-wise Int4/Int8，手写 SIMD 内核

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Rust 2021 Edition |
| 版本 | 0.11.0 |
| 后端 | gllm-kernels（本地路径依赖） |
| 模型加载 | hf-hub, safetensors, memmap2 |
| Tokenizer | tokenizers |
| 序列化 | serde, serde_json, prost |
| 并行 | rayon |
| 异步（可选） | tokio |

## 核心设计原则

1. **准确度优先 > 吞吐量** - 绝不为调度优化牺牲计算精度
2. **严格因果顺序** - 批内 attention 计算必须保证严格因果掩码
3. **可靠性优先** - 内存管理有严格边界检查和错误恢复
4. **确定性调度** - 优先确定性串行执行
5. **融合优先** - 优先选择融合算子

## 支持的模型

| 类别 | 模型 | 架构 |
|------|------|------|
| Generator | qwen3-7b, qwen3-moe, llama-4-8b, glm-5-9b, mistral-small-3 | Dense/MoE |
| Embedding | qwen3-embed, bge-m4 | Transformer |
| Rerank | qwen3-rerank, bge-rerank-v3 | XLM-R-Next |

## 后端支持

- **CUDA** (NVIDIA): libcuda.so 直接加载，AOT 内核 sm_80/86/89/90
- **Metal** (Apple): 原生 Metal.framework
- **ROCm** (AMD): libhsa-runtime64.so 直接加载
- **CPU** (Fallback): faer（AVX-512/NEON）纯 Rust SIMD
