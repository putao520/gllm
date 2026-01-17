# gllm 功能需求规格

## 概述

gllm 是一个纯 Rust 实现的本地嵌入和重排序推理库，基于 gllm-kernels 的零成本算子与权重容器，提供 OpenAI 风格的 SDK API，支持静态编译。

## 修订历史

| 版本 | 日期 | 描述 |
|------|------|------|
| v0.2.0 | 2025-11-28 | 完整E2E测试覆盖26个模型 - 新增中文模型支持和下载验证 |
| v0.1.0 | 2025-01-28 | 初始版本 |

---

## 核心需求

### REQ-CORE-001: 纯 Rust 实现

**描述**: 整个库必须使用纯 Rust 实现，支持静态编译

**验收标准**:
- 可通过 `cargo build --release --target x86_64-unknown-linux-musl` 静态编译
- 无 C/C++ 依赖
- 所有依赖库均为纯 Rust 或提供纯 Rust 特性

**状态**: ✅ 已实现 [PRD-03]

---

## 模型管理

### REQ-MODEL-001: 自动模型下载

**描述**: 支持从 HuggingFace 自动下载指定模型

**验收标准**:
- 首次使用模型时自动从 HF 下载
- 模型存储在 `~/.gllm/models/` 目录
- 支持 SafeTensors 格式
- 使用 rustls 作为 TLS 后端

**状态**: ✅ 已实现 [PRD-01]

### REQ-MODEL-002: 模型别名系统

**描述**: 提供简化的模型别名，映射到 HuggingFace repo ID

**验收标准**:
- 支持 `qwen2.5:7b` 风格的简化别名
- 内置常用模型别名注册表
- 也支持直接使用 HF repo ID

**状态**: ✅ 已实现 [PRD-01]

### REQ-MODEL-003: SafeTensors 加载

**描述**: 加载 SafeTensors 格式模型权重到 WeightMatrix/Vector

**验收标准**:
- 使用 safetensors + WeightLoader 解析权重
- 支持 FP16/FP32 精度
- 支持量化模型加载

**状态**: ✅ 已实现 [PRD-01]

---

## 推理功能

### REQ-INFER-001: Embedding 推理

**描述**: 支持文本嵌入向量生成

**验收标准**:
- 支持 BERT 架构 (基于 gllm-kernels 零成本算子实现)
- 返回 Vec<f32> 向量
- 支持批量输入

**状态**: ✅ 已实现 [PRD-02]

### REQ-INFER-002: Rerank 推理

**描述**: 支持文档重排序

**验收标准**:
- 支持 Cross-Encoder 架构 (基于 gllm-kernels 零成本算子实现)
- 输入 query + documents，输出排序后的文档列表
- 返回相关性分数

**状态**: ✅ 已实现 [PRD-02]

### REQ-INFER-003: Generator 推理

**描述**: 支持文本生成（LLM 推理）

**验收标准**:
- 支持多种 Decoder 架构（详见 05-MODEL-REGISTRY.md）：
  - Qwen2Generator: Qwen2/2.5 系列
  - Qwen3Generator: Qwen3 系列
  - Qwen3MoE: Qwen3 MoE 系列
  - MistralGenerator: Mistral 系列
  - Mixtral: Mixtral MoE 系列
  - Phi3Generator: Phi-4 系列
  - SmolLM3Generator: SmolLM3 系列
  - InternLM3Generator: InternLM3 系列
  - GLM4: GLM-4 系列
  - GLM4MoE: GLM-4.7 MoE 系列
  - DeepSeekV3: DeepSeek-V3 系列
- 支持 FP16/GGUF/AWQ 等量化格式
- 支持 KV Cache 增量推理
- 返回生成的 token 序列

**状态**: ✅ 已实现 (2025-01-17)

---

## API 设计

### REQ-API-001: OpenAI 风格 SDK

**描述**: 提供类似 OpenAI SDK 风格的 Rust API

**验收标准**:
- Builder 模式构建请求
- 类型安全的请求/响应结构
- 符合 Rust 惯用法

**状态**: ✅ 已实现 [PRD-02]

### REQ-API-002: 同步 API

**描述**: 提供同步调用接口

**验收标准**:
- `Client::new()` 同步初始化
- `.generate()` 同步返回完整结果

**状态**: ✅ 已实现 [PRD-02]

### REQ-API-003: 异步 API

**描述**: 提供异步调用接口

**验收标准**:
- `AsyncClient::new().await` 异步初始化
- `.generate().await` 异步返回
- 可选特性，通过 feature flag 启用

**状态**: ✅ 已实现 [PRD-02]

---

## 后端支持

### REQ-BACKEND-001: WGPU 后端

**描述**: 支持 WGPU 作为 GPU 后端

**验收标准**:
- 纯 Rust 实现
- 跨平台支持 (Vulkan/DX12/Metal)
- 运行时后端检测中自动选择或回退

**状态**: ✅ 已实现 [PRD-02]

### REQ-BACKEND-002: CPU 后端

**描述**: 支持 gllm-kernels CPU 后端

**验收标准**:
- 纯 Rust 实现
- 无 GPU 环境可用时自动回退
- 与运行时后端检测兼容

**状态**: ✅ 已实现 [PRD-02]

---

## 量化支持

### REQ-QUANT-001: 模型量化

**描述**: 支持量化模型以减少内存占用

**验收标准**:
- 支持 gllm-kernels 量化算子与 AWQ/GGUF 权重
- 支持 INT8/INT4 量化
- 保持纯 Rust 实现

**状态**: 🔄 基础支持已实现 [PRD-02]（通过 SafeTensors 加载支持量化模型，但未实现专用量化推理优化）

---

## gllm-kernels 集成

### REQ-KERN-001: 运行时后端选择

**描述**: 使用 gllm-kernels 实现运行时 GPU 后端自动检测和选择

**验收标准**:
- 程序启动时自动检测可用后端（CUDA > ROCm > Metal > WGPU > CPU）
- 支持 `GLLM_BACKEND` 环境变量强制指定后端
- 同一二进制支持所有 GPU 厂商，用户无需重新编译

**关联设计**: ARCH-ADR-007

**状态**: ✅ 已实现 (2025-01-17) - KernelDispatcher 自动检测后端

### REQ-KERN-002: 2M 超长上下文支持

**描述**: Attention 计算支持 2M+ token 上下文，无数值溢出

**验收标准**:
- 使用 gllm-kernels 的 LogSpaceSoftmax 防止 exp 溢出
- 使用 KahanAccumulator 减少浮点累积误差
- 支持增量计算，无需一次性加载全序列

**关联设计**: ARCH-ADR-008

**状态**: ✅ 已实现 (2025-01-17) - flash_attention 实现

### REQ-KERN-003: 零成本算子调用

**描述**: Attention 算子调用必须是零成本抽象

**验收标准**:
- 使用原生切片 `&[f16]` 作为算子输入，避免额外抽象开销
- 推理使用 gllm-kernels 原生算子与 WeightMatrix/Vector
- 无 trait object 动态派发

**关联设计**: ARCH-ADR-007

**状态**: ✅ 已实现 (2025-01-17) - Burn 依赖完全移除，纯 gllm-kernels 实现
