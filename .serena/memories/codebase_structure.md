# gllm 代码结构

## 目录结构

```
src/
├── lib.rs              # 库入口，导出所有公共模块
├── client.rs           # Client/AsyncClient API 入口
├── embeddings.rs       # 嵌入生成
├── generation.rs       # 文本生成
├── rerank.rs           # 重排序
├── tokenizer.rs        # Tokenizer 封装
├── model_config.rs     # 模型配置
├── quantization.rs     # 量化支持
├── weight_loader.rs    # 权重加载
├── kv_cache.rs         # KV 缓存
│
├── adapter/            # 模型适配器（架构特定实现）
│   ├── mod.rs
│   ├── trait.rs        # ModelAdapter trait
│   ├── qwen3.rs        # Qwen3 适配器
│   ├── qwen3_moe.rs    # Qwen3 MoE
│   ├── qwen3_embed.rs  # Qwen3 Embedding
│   ├── qwen3_rerank.rs # Qwen3 Rerank
│   ├── llama4.rs       # Llama 4
│   ├── glm5.rs         # GLM-5
│   ├── mistral3.rs     # Mistral 3
│   ├── ministral.rs    # Ministral
│   ├── gemma2.rs       # Gemma 2
│   ├── phi4.rs         # Phi 4
│   ├── qwen2_5.rs      # Qwen 2.5
│   ├── gpt_oss.rs      # GPT OSS
│   └── xlm_r.rs        # XLM-RoBERTa
│
├── backend/            # 硬件后端
│   ├── mod.rs
│   ├── detection.rs    # 硬件自动检测
│   └── fallback.rs     # CPU 回退
│
├── engine/             # 执行引擎
│   ├── mod.rs
│   └── executor.rs     # 执行器（封装 gllm-kernels）
│
├── loader/             # 模型加载器
│   ├── mod.rs
│   ├── adapter.rs      # 加载适配
│   ├── config.rs       # 配置加载
│   ├── downloader.rs   # 下载管理
│   ├── format_detector.rs # 格式检测
│   ├── hf_hub.rs       # HuggingFace Hub
│   ├── modelscope.rs   # ModelScope
│   ├── parallel.rs     # 并行加载
│   ├── pytorch.rs      # PyTorch 格式
│   ├── safetensors.rs  # SafeTensors 格式
│   ├── gguf/           # GGUF 格式支持
│   └── onnx/           # ONNX 格式支持
│       ├── matcher/    # 图模式匹配
│       └── tensor/     # 张量解析
│
├── scheduler/          # 调度器
│   ├── mod.rs
│   ├── paged_scheduler.rs  # PagedAttention 调度
│   ├── allocator.rs    # 内存分配
│   ├── batcher.rs      # 连续批处理
│   ├── memory_manager.rs # 内存管理
│   ├── sequence.rs     # 序列管理
│   ├── types.rs        # 类型定义
│   ├── vllm2024.rs     # vLLM 2024 兼容
│   └── ...
│
├── manifest/           # 模型清单
│   ├── mod.rs
│   └── types.rs
│
└── bin/
    ├── download.rs     # 下载工具
    └── debug_shape.rs  # 调试工具

tests/                  # 测试文件
├── common.rs           # 公共测试工具
├── test_e2e_*.rs       # E2E 测试
└── test_*.rs           # 单元/集成测试

SPEC/                   # 规格文档
├── 01-REQUIREMENTS.md
├── 02-ARCHITECTURE.md
├── 03-DATA-STRUCTURE.md
├── 04-API-DESIGN.md
├── 06-TESTING-STRATEGY.md
└── 07-OBSERVABILITY.md
```

## 核心模块

- **Client**: 用户 API 入口（new_chat/new_embedding/generate/embeddings/rerank）
- **adapter**: 模型架构适配（每个模型一个适配器）
- **loader**: 多源加载（HF/ModelScope/SafeTensors/GGUF/ONNX）
- **engine**: 执行引擎（封装 gllm-kernels L3 API）
- **scheduler**: PagedAttention 和连续批处理
