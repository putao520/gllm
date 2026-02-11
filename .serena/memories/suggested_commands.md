# gllm 常用命令

## 构建命令

```bash
# 检查编译
cargo check

# 构建（debug）
cargo build

# 构建（release）
cargo build --release

# 构建带特定 feature
cargo build --features tokio
cargo build --features candle
cargo build --features paged-attention
```

## 测试命令

```bash
# 运行所有单元测试（可并行）
cargo test --lib

# 运行所有测试
cargo test

# 🚨 E2E 测试必须单线程运行
cargo test --test test_e2e_embedding -- --test-threads=1
cargo test --test test_e2e_generator -- --test-threads=1
cargo test --test test_e2e_reranker -- --test-threads=1

# 运行特定测试
cargo test test_name

# 运行测试并显示输出
cargo test -- --nocapture
```

## 格式化和 Lint

```bash
# 格式化代码
cargo fmt

# 检查格式
cargo fmt --check

# Clippy lint
cargo clippy
cargo clippy --all-features

# Clippy 修复
cargo clippy --fix
```

## 文档

```bash
# 生成文档
cargo doc

# 生成并打开文档
cargo doc --open
```

## 二进制工具

```bash
# 下载模型
cargo run --bin download -- <model_id>

# 调试形状
cargo run --bin debug_shape -- <args>
```

## 环境变量

```bash
# 自定义缓存目录
export GLLM_CACHE_DIR=~/.gllm/models

# HuggingFace Token
export HF_TOKEN=<your_token>

# 日志级别
export RUST_LOG=gllm=debug
```

## Git 命令

```bash
git status
git add .
git commit -m "feat: description"
git push
git pull
git log --oneline -10
```

## 系统命令（Linux）

```bash
ls -la
cd <dir>
find . -name "*.rs"
grep -r "pattern" src/
cat <file>
head -n 50 <file>
tail -n 50 <file>
```
