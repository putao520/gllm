# gllm 测试环境配置

## 概述

gllm 是 Rust 库，测试环境相对简单，主要是验证 API 功能和模型推理逻辑。

## 测试环境架构

### 依赖环境
- **主机环境**: 测试在开发主机直接运行（不是容器化服务）
- **网络访问**: 某些测试需要访问 HuggingFace API（但会避免真实下载）
- **文件系统**: 需要 `~/.gllm/models/` 目录访问权限

### 测试隔离
- 使用临时目录模拟模型存储
- 使用虚假 SafeTensors 文件进行测试
- 每个测试后自动清理临时文件

## 环境变量

测试可能使用的环境变量：

```bash
# 模型存储目录覆盖（用于测试隔离）
GLLM_MODEL_DIR=/tmp/gllm-test-<uuid>/models

# 跳过真实网络下载（测试模式）
GLLM_TEST_MODE=1

# 测试超时控制
GLLM_TEST_TIMEOUT=30  # 秒
```

## 测试基础设施

### 已有测试工具
- **测试框架**: cargo test (Rust 内置)
- **断言库**: 标准 assert! 宏
- **Mock/Factory**: 自定义测试辅助函数

### 测试数据策略
1. **SafeTensors 文件**: 使用 `write_dummy_weights()` 生成假模型文件
2. **Tokenizer**: 使用临时测试 tokenizer 文件
3. **临时目录**: 使用 tempfile crate 创建和清理

### 测试服务
- **无外部依赖**: 不依赖 MongoDB, Redis 等外部服务
- **纯本地测试**: 所有测试在单个进程内完成

## 测试命令

```bash
# 运行所有测试
cargo test

# 运行特定功能测试
cargo test embeddings
cargo test rerank

# 运行集成测试
cargo test --test api

# 带详细输出
cargo test -- --nocapture

# 覆盖率（需要安装 cargo-tarpaulin）
cargo tarpaulin --out Html
```

## Feature Flag 测试矩阵

需要测试的 feature 组合：

| Feature组合 | 命令 | 说明 |
|------------|------|------|
| wgpu (默认) | `cargo test` | 默认 WGPU 后端 |
| cpu | `cargo test --features cpu` | CPU 后端 |
| async | `cargo test --features "wgpu,async"` | 异步 API |
| cpu+async | `cargo test --features "cpu,async"` | CPU + 异步 |

## 测试清理策略

- **临时文件**: 使用 tempfile 自动清理
- **模型目录**: 测试后删除创建的临时模型目录
- **网络请求**: Mock HF Hub 响应，避免真实下载